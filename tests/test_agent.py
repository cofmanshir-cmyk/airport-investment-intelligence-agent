"""Agent loop checks using a scripted client, so no API key or network is needed.

The scripted client stands in for the LLM in tests only; the application always
uses the real API.
"""

from types import SimpleNamespace

import pytest

from src import cli, config, presentation, tools
from src.agent import (
    Agent,
    AgentError,
    Answer,
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    ToolCall,
    find_unverified_numbers,
    main_set,
    primary_ranking,
    split_primary_table,
)

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "ontime_airport_month.parquet").exists(),
    reason="Processed data not built",
)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def reply(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        next_reply = self.replies.pop(0)
        if isinstance(next_reply, Exception):
            raise next_reply
        return next_reply


def test_tool_result_comes_from_python_and_returns_to_the_llm():
    client = ScriptedClient(
        reply("tool_use", tool_block("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000})),
        reply("end_turn", text_block("Scheduled passenger flights: 2.6% long-haul.")),
    )
    answer = Agent(client).ask("What share of ANC flights are long-haul?")

    assert answer.text == "Scheduled passenger flights: 2.6% long-haul."
    assert answer.tool_calls[0].result["scopes"]["scheduled_passenger"]["share_pct"] == 2.6
    assert answer.unverified_numbers == []
    returned = client.requests[1]["messages"][-1]["content"][0]
    assert returned["tool_use_id"] == "toolu_1"
    assert returned["is_error"] is False
    assert '"share_pct": 2.6' in returned["content"]


def tool_call(name, result, is_error=False):
    return ToolCall(name, {}, result, is_error)


NEW_ENGLAND = ["CT", "MA", "ME", "NH", "RI", "VT"]


def test_score_ranking_defines_the_main_set_whatever_the_call_order():
    full = [row["airport"] for row in tools.rank_opportunity_score(NEW_ENGLAND, 25)["ranked"]]
    score = tools.rank_opportunity_score(NEW_ENGLAND, 5)
    expected = full[:5]
    outside = full[5:7]
    assert [row["airport"] for row in score["ranked"]] == expected and len(outside) == 2
    growth = tools.rank_airports(3, NEW_ENGLAND, 10, 100_000)
    profile = tools.get_airport_profile(outside[0])
    compared = tools.compare_airports(outside, ["congestion"])
    calls = [tool_call("rank_opportunity_score", score), tool_call("rank_airports", growth), tool_call("get_airport_profile", profile), tool_call("compare_airports", compared)]
    assert main_set(calls) == expected
    assert main_set(list(reversed(calls))) == expected
    assert not set(outside) & set(main_set(calls))


def test_growth_ranking_is_the_main_set_when_no_score_ranking_has_rows():
    growth = tools.rank_airports(3, NEW_ENGLAND, 3, 100_000)
    empty_score = tools.rank_opportunity_score(["DE"], 1, 1_000_000_000)
    rejected_score = tool_call("rank_opportunity_score", {"error": "Bad"}, True)
    expected = [row["airport"] for row in growth["ranked"]]
    assert main_set([tool_call("rank_airports", growth)]) == expected
    assert main_set([tool_call("rank_opportunity_score", empty_score), tool_call("rank_airports", growth)]) == expected
    assert main_set([rejected_score, tool_call("rank_airports", growth)]) == expected


def test_main_set_uses_comparisons_then_single_airports_and_ignores_rejections():
    compared = tool_call("compare_airports", tools.compare_airports(["LAX", "SNA"]))
    more = tool_call("compare_airports", tools.compare_airports(["SNA", "SFO"]))
    profile = tool_call("get_airport_profile", tools.get_airport_profile("BOS"))
    long_haul = tool_call("get_long_haul_share", tools.get_long_haul_share("ANC", 3000))
    rejected = tool_call("rank_airports", {"error": "Bad"}, True)
    assert main_set([profile, compared, more]) == ["LAX", "SNA", "SFO"]
    assert main_set([profile, long_haul, profile]) == ["BOS", "ANC"]
    assert main_set([rejected]) == []
    assert main_set([]) == []


def test_ranking_with_no_rows_does_not_define_the_set():
    empty = tools.rank_opportunity_score(["DE"], 1, 1_000_000_000)
    assert empty["ranked"] == []
    profile = tool_call("get_airport_profile", tools.get_airport_profile("BOS"))
    assert main_set([tool_call("rank_opportunity_score", empty), profile]) == ["BOS"]


def test_command_line_agent_also_sends_the_main_set():
    client = ScriptedClient(
        reply("tool_use", tool_block("rank_opportunity_score", {"states": [], "top_n": 3})),
        reply("end_turn", text_block("Shortlist.")),
        reply("end_turn", text_block("Done.")),
    )
    agent = Agent(client)
    agent.ask("Start?")
    agent.ask("Which of them?")
    expected = ", ".join(row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"])
    assert client.requests[0]["messages"][0]["content"] == "Start?"
    assert client.requests[2]["messages"][-1]["content"].endswith(f"was {expected}.)")


def test_score_table_is_the_main_table_whatever_the_call_order():
    score = tool_call("rank_opportunity_score", tools.rank_opportunity_score([], 3))
    growth = tool_call("rank_airports", tools.rank_airports(3, [], 3, 100_000))
    other_window = tool_call("rank_airports", tools.rank_airports(10, [], 3, 100_000))
    compared = tool_call("compare_airports", tools.compare_airports(["LAX", "SNA"], ["congestion"]))
    assert [table.title for table in presentation.main_tables([score, growth, compared])] == [
        "Airport Opportunity Score", "Operational congestion",
    ]
    assert [table.title for table in presentation.main_tables([growth, score])] == ["Airport Opportunity Score"]
    assert [table.title for table in presentation.main_tables([growth, other_window])] == [
        "Ranked by Passenger CAGR 3Y", "Ranked by Passenger CAGR 10Y",
    ]


def test_growth_table_stays_main_when_the_score_ranking_has_no_rows():
    empty_score = tool_call("rank_opportunity_score", tools.rank_opportunity_score(["DE"], 1, 1_000_000_000))
    growth = tool_call("rank_airports", tools.rank_airports(3, [], 3, 100_000))
    assert [table.title for table in presentation.main_tables([empty_score, growth])] == ["Ranked by Passenger CAGR 3Y"]


def test_screening_guidance_reaches_the_model_without_forcing_a_tool():
    prompt = SYSTEM_PROMPT
    assert "call rank_opportunity_score first as the primary ranking" in prompt
    assert "Use rank_airports when the question is about growth itself" in prompt
    assert "Set top_n to the number of airports you will discuss" in prompt
    assert "starting points for further investigation, never as confirmed needs" in prompt
    assert "do not claim that every top-ranked airport needs action" in prompt
    assert "say so and give the reason using tool values" in prompt
    assert "If a question uses a label the data cannot confirm" in prompt
    assert "[primary_table: rank_airports]" in prompt
    assert "80 to 120 words" in prompt and "no headings, bullet lists or labels" in prompt
    assert "use relevant indicators" not in prompt
    source = (config.ROOT / "src" / "agent.py").read_text() + (config.ROOT / "src" / "gemini_agent.py").read_text()
    assert "tool_choice" not in source and "allowed_function_names" not in source


def test_main_table_line_is_split_from_the_text():
    assert split_primary_table("Answer.\n\n[primary_table: rank_airports]") == ("Answer.", "rank_airports")
    assert split_primary_table("Answer.\n[Primary Table: rank_airports]") == ("Answer.", "rank_airports")
    assert split_primary_table("Answer.\n[primary_table: abc]") == ("Answer.", "ABC")
    assert split_primary_table("Answer.\n[primary_table: ]") == ("Answer.", None)
    assert split_primary_table("Answer with [brackets].") == ("Answer with [brackets].", None)


def test_named_ranking_overrides_the_default_only_when_it_has_rows():
    score = tool_call("rank_opportunity_score", tools.rank_opportunity_score(NEW_ENGLAND, 3))
    growth = tool_call("rank_airports", tools.rank_airports(3, NEW_ENGLAND, 3, 100_000))
    empty_growth = tool_call("rank_airports", tools.rank_airports(3, ["DE"], 1, 250_000, min_passengers_2025=1_000_000_000))
    assert empty_growth.result["ranked"] == []
    assert primary_ranking([score, growth]) is score
    assert primary_ranking([score, growth], "rank_airports") is growth
    assert primary_ranking([score, empty_growth], "rank_airports") is score
    assert [table.title for table in presentation.main_tables([score, growth], "rank_airports")] == ["Ranked by Passenger CAGR 3Y"]


def test_command_line_agent_follows_the_main_table_line():
    client = ScriptedClient(
        reply("tool_use", tool_block("rank_opportunity_score", {"states": [], "top_n": 3}, "a"), tool_block("rank_airports", {"window_years": 3, "states": [], "top_n": 3, "min_enplanements": 100_000}, "b")),
        reply("end_turn", text_block("Growth answer.\n[primary_table: rank_airports]")),
    )
    answer = Agent(client).ask("How fast?")
    assert answer.text == "Growth answer."
    rendered = cli.render_answer(answer)
    assert "Ranked by Passenger CAGR 3Y" in rendered and "primary_table" not in rendered
    assert "Airport Opportunity Score\n" not in rendered


def test_guidance_asks_for_a_reasoning_analyst_without_a_report_structure():
    assert 'Say "I\'d", take a position' in SYSTEM_PROMPT
    assert "what could make it misleading" in SYSTEM_PROMPT
    assert "Build uncertainty and scope into your sentences" in SYSTEM_PROMPT
    assert "When asked to choose, choose one and say why" in SYSTEM_PROMPT
    assert "cite only the few tool numbers that carry your reasoning" in SYSTEM_PROMPT
    assert 'starting with "Warning:"' not in SYSTEM_PROMPT


def test_guidance_answers_first_and_keeps_limits_for_the_conclusion():
    assert "Answer the business question first" in SYSTEM_PROMPT
    assert "Never open with methodology, data limits or what the tools cannot measure" in SYSTEM_PROMPT
    assert "your first sentence names the airports you would look at" in SYSTEM_PROMPT
    assert "weaving in what this does not establish" in SYSTEM_PROMPT
    assert "Then say what makes each one stand out" in SYSTEM_PROMPT
    # The first-sentence limit now applies only when the question asks for a value no tool has.
    assert "something no tool measures directly, say so in your first sentence" not in SYSTEM_PROMPT
    assert "Only when the question asks for a value no tool measures" in SYSTEM_PROMPT
    assert "Never give an amount for the unmeasured thing" in SYSTEM_PROMPT


def test_guidance_requires_the_scope_of_partial_figures():
    assert "say which flights, along with the threshold, period and carrier scope" in SYSTEM_PROMPT
    assert "give any range it returns" in SYSTEM_PROMPT
    assert "Never present a share of all flights as a share of passenger flights" in SYSTEM_PROMPT


def test_number_check_limits_are_documented_and_real():
    # These pass because the check only asks whether a value appears anywhere in the evidence.
    evidence = [tools.to_json(tools.get_airport_profile("SFO"))]
    cagr = tools.get_airport_profile("SFO")["profile"]["demand_pressure_proxies"]["metrics"]["passenger_cagr_3y_pct"]
    assert find_unverified_numbers(f"Taxi-out at SFO is {cagr} minutes.", evidence) == []
    assert find_unverified_numbers("SFO has 12 gates.", evidence) == []
    assert "limited safeguard" in find_unverified_numbers.__doc__
    assert "limited safeguard" in (config.ROOT / "README.md").read_text()


def test_guidance_stays_general_and_contains_no_canned_answers():
    for phrase in (
        "I'd start with", "shortlist rather than", "put a reliable number", "the interesting part",
        "for different reasons", "master plan", "gate capacity", "promising leads",
    ):
        assert phrase.lower() not in SYSTEM_PROMPT.lower()


def test_guidance_asks_for_batched_look_ups_and_a_table_only_for_what_is_featured():
    assert "request them together with compare_airports or several calls in the same turn" in SYSTEM_PROMPT
    assert "use empty table_sections for look-ups" in SYSTEM_PROMPT
    assert "show no table or one compact table for that airport" in SYSTEM_PROMPT


def test_command_line_answer_shows_only_the_featured_airport_table():
    first, second = [row["airport"] for row in tools.rank_opportunity_score([], 2)["ranked"]]
    calls = [
        ToolCall("get_airport_profile", {}, tools.get_airport_profile(code, ["score"]), False)
        for code in (first, second)
    ]
    rendered = cli.render_answer(Answer(f"Start with {second}.", calls))
    table_header = rendered.split("Airport Opportunity Score", 1)[1].split("\n")[1]
    assert second in table_header and first not in table_header


def test_tool_descriptions_separate_screening_from_growth_and_schemas_stay_the_same():
    definitions = {tool["name"]: tool for tool in tools.TOOL_DEFINITIONS}
    assert "Use this first when the user asks which airports look most promising overall" in definitions["rank_opportunity_score"]["description"]
    assert "not for overall investment or expansion screening" in definitions["rank_airports"]["description"]
    assert list(definitions["rank_opportunity_score"]["input_schema"]["properties"]) == ["states", "top_n", "min_passengers_2025", "passenger_comparison"]
    assert list(definitions["rank_airports"]["input_schema"]["properties"]) == [
        "window_years", "states", "top_n", "min_enplanements", "rank_by", "min_passengers_2025", "passenger_comparison",
    ]


def test_command_line_shows_the_long_haul_table_and_its_scope():
    call = ToolCall("get_long_haul_share", {}, tools.get_long_haul_share("ANC", 3000), False)
    rendered = cli.render_answer(Answer("ANC figures.", [call]))
    assert "Passenger aircraft" in rendered and "Freighters" in rendered
    assert "3,000 statute miles" in rendered and "January\u2013July 2026, US carriers only" in rendered


def test_command_line_evidence_shows_score_disclaimer_and_small_base_warning():
    result = tools.rank_opportunity_score(["DE"], 1)
    answer = Answer("ILG scored 78.38.", [ToolCall("rank_opportunity_score", {"states": ["DE"], "top_n": 1}, result, False)])
    text = cli.render_evidence(answer)
    assert "Preliminary screening proxy" in text
    assert "WARNING: ILG: Small base year: 2022 had 4,988 passengers" in text


SAMPLE_ANSWER = """ILG, DRO and SAF rank highest on the score among all eligible airports.

These are airports worth further investigation, not confirmed terminal needs. ILG's score reflects very fast growth \
from almost no traffic, while DRO and SAF combine strong percentage growth with high load factors. Larger airports \
add far more passengers, so absolute growth and CAGR tell different stories.

Warning: ILG has a small base, and terminal capacity is not directly measured."""


def narrative_words(text):
    return len(" ".join(line for line in text.splitlines() if not line.lstrip().startswith("|")).split())


def test_concise_sample_stays_within_the_word_target_and_keeps_tables_out_of_the_narrative():
    assert narrative_words(SAMPLE_ANSWER) <= 120
    conclusion, rest = presentation.split_answer(SAMPLE_ANSWER)
    assert conclusion.endswith("eligible airports.")
    assert rest.startswith("These are airports")
    assert rest.count("Warning:") == 1


def test_command_line_answer_places_the_python_table_after_the_conclusion():
    result = tools.rank_opportunity_score([], 3)
    answer = Answer(SAMPLE_ANSWER, [ToolCall("rank_opportunity_score", {"states": [], "top_n": 3}, result, False)])
    text = cli.render_answer(answer)
    assert text.index("eligible airports.") < text.index("#1 of 229") < text.index("These are airports")
    assert "78.38" in text
    assert "WARNING: Small base year" not in text


def test_command_line_answer_shows_a_notice_when_the_warning_is_missing():
    result = tools.rank_opportunity_score(["DE"], 1)
    answer = Answer("ILG leads.", [ToolCall("rank_opportunity_score", {"states": ["DE"], "top_n": 1}, result, False)])
    assert "WARNING: Small base year: ILG (" in cli.render_answer(answer)


def test_model_written_table_is_removed_from_the_narrative():
    conclusion, rest = presentation.split_answer("Top.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nInsight.")
    assert (conclusion, rest) == ("Top.", "Insight.")


def test_absolute_ranking_lists_the_ranked_measure_first():
    result = tools.rank_airports(3, [], 2, 100_000, "absolute_passenger_growth")
    table = presentation.main_tables([ToolCall("rank_airports", {}, result, False)])[0].data
    columns = list(table.columns)
    assert columns.index("Absolute Passenger Growth 3Y") < columns.index("Passenger CAGR 3Y (%)")
    assert table.iloc[0]["National rank"].startswith("#1 of ")


def test_cagr_ranking_lists_the_ranked_measure_first():
    result = tools.rank_airports(3, [], 2, 100_000)
    columns = list(presentation.main_tables([ToolCall("rank_airports", {}, result, False)])[0].data.columns)
    assert columns.index("Passenger CAGR 3Y (%)") < columns.index("Absolute Passenger Growth 3Y")


def test_tool_error_produces_no_results_table():
    assert presentation.main_tables([ToolCall("rank_airports", {}, {"error": "Bad"}, True)]) == []


def test_agent_code_does_not_route_by_question_words():
    source = (config.ROOT / "src" / "agent.py").read_text() + (config.ROOT / "src" / "gemini_agent.py").read_text()
    assert "question.lower" not in source
    assert " in question" not in source
    assert "tool_choice" not in source


def test_invented_number_is_flagged():
    client = ScriptedClient(
        reply("tool_use", tool_block("get_airport_profile", {"airport": "SFO"})),
        reply("end_turn", text_block("SFO grew 8.87% a year and has 1,234,567 unmet passengers.")),
    )
    assert Agent(client).ask("Tell me about SFO").unverified_numbers == ["1,234,567"]


def test_answer_without_a_tool_flags_every_number():
    client = ScriptedClient(reply("end_turn", text_block("LAX has 87.3 million passengers.")))
    answer = Agent(client).ask("How big is LAX?")
    assert answer.tool_calls == []
    assert answer.unverified_numbers == ["87.3 million"]


def test_follow_up_keeps_history():
    client = ScriptedClient(
        reply("tool_use", tool_block("get_airport_profile", {"airport": "SFO"})),
        reply("end_turn", text_block("First answer.")),
        reply("end_turn", text_block("Second answer.")),
    )
    agent = Agent(client)
    agent.ask("Tell me about SFO")
    agent.ask("And its taxi-out?")
    roles = [message["role"] for message in client.requests[2]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_bad_tool_input_returns_an_error_the_llm_can_correct():
    client = ScriptedClient(
        reply("tool_use", tool_block("get_airport_profile", {"airport": "ZZZZ"})),
        reply("end_turn", text_block("I could not find that airport.")),
    )
    answer = Agent(client).ask("Tell me about ZZZZ")
    assert answer.tool_calls[0].is_error
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_parallel_tool_calls_return_in_one_message():
    client = ScriptedClient(
        reply(
            "tool_use",
            tool_block("get_airport_profile", {"airport": "SFO"}, "toolu_a"),
            tool_block("get_airport_profile", {"airport": "LAX"}, "toolu_b"),
        ),
        reply("end_turn", text_block("Done.")),
    )
    Agent(client).ask("Profiles for SFO and LAX")
    results = client.requests[1]["messages"][-1]["content"]
    assert [result["tool_use_id"] for result in results] == ["toolu_a", "toolu_b"]


def test_refusal_raises_and_leaves_history_clean():
    client = ScriptedClient(reply("refusal"), reply("end_turn", text_block("Recovered.")))
    agent = Agent(client)
    with pytest.raises(AgentError):
        agent.ask("Question")
    assert agent.messages == []
    assert agent.ask("Another question").text == "Recovered."


def test_failed_api_call_leaves_history_clean():
    client = ScriptedClient(RuntimeError("network"), reply("end_turn", text_block("OK.")))
    agent = Agent(client)
    with pytest.raises(RuntimeError):
        agent.ask("Question")
    assert agent.messages == []


def test_max_tokens_is_reported():
    with pytest.raises(AgentError, match="length"):
        Agent(ScriptedClient(reply("max_tokens", text_block("Partial")))).ask("Question")


def test_pause_turn_continues():
    client = ScriptedClient(reply("pause_turn"), reply("end_turn", text_block("Finished.")))
    assert Agent(client).ask("Question").text == "Finished."


def test_endless_tool_calls_stop_at_the_limit():
    replies = [
        reply("tool_use", tool_block("get_airport_profile", {"airport": "SFO"})) for _ in range(MAX_TOOL_ROUNDS)
    ]
    with pytest.raises(AgentError, match="tool rounds"):
        Agent(ScriptedClient(*replies)).ask("Question")


def test_request_uses_tools_and_never_forces_one():
    client = ScriptedClient(reply("end_turn", text_block("Hello.")))
    Agent(client).ask("Hello")
    request = client.requests[0]
    assert request["tools"] == tools.TOOL_DEFINITIONS
    assert request["model"]
    assert "tool_choice" not in request


@pytest.mark.parametrize(
    ("answer", "evidence", "expected"),
    [
        ("Growth was 8.87% a year.", ["8.866451"], []),
        ("It carried 26.5 million passengers.", ["26488386"], []),
        ("It carried 26,488,386 passengers.", ["26488386"], []),
        ("It fell by -0.41 points.", ["-0.41"], []),
        ("Over 3 years in 2025, the top 5.", [], []),
        ("Growth was 9.99%.", ["8.87"], ["9.99%"]),
        ("It carried 30 million passengers.", ["26488386"], ["30 million"]),
    ],
)
def test_number_check(answer, evidence, expected):
    assert find_unverified_numbers(answer, evidence) == expected


def test_command_line_agent_reset_starts_a_new_conversation():
    client = ScriptedClient(
        reply("tool_use", tool_block("rank_opportunity_score", {"states": [], "top_n": 3})),
        reply("end_turn", text_block("Shortlist.")),
        reply("end_turn", text_block("Fresh.")),
    )
    agent = Agent(client)
    agent.ask("Start?")
    agent.reset()
    agent.ask("New question")
    assert client.requests[-1]["messages"][0]["content"] == "New question"
    assert len(client.requests[-1]["messages"]) == 1
