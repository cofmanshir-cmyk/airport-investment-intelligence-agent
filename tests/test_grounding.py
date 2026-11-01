"""Grounding safeguard checks, using the real Gemini SDK over a fake network.

Each script stands in for one possible run of the model, so a skipped tool call is deterministic here.
No request leaves this machine and no API allowance is used.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from src import config, gemini_agent, grounding, tools
from src.agent import AgentError, ToolCall
from src.gemini_agent import GeminiAgent

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "t100_airport_month.parquet").exists(),
    reason="Processed data not built",
)

QUESTION = "What is the unmet flight demand in SFO airport and why?"
DRAFT = "Unmet demand there is about 4 million passengers because the runways are full."
INDIRECT = "Unmet demand can't be measured directly here, so I'd look at growth and load factor as indirect signs."
PROFILE = {"airport": "SFO", "table_sections": []}


def function_call(name, args):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"id": "call_1", "name": name, "args": args}}
    ]}, "finishReason": "STOP"}]}


def text_reply(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}


def profile_call(code):
    return function_call("get_airport_profile", {"airport": code, "table_sections": []})


class FakeNetwork:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, request):
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))

    @property
    def modes(self):
        return [body["toolConfig"]["functionCallingConfig"]["mode"] for body in self.bodies]


def make_agent(network):
    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(network)),
        retry_options=types.HttpRetryOptions(attempts=1),
    ))
    return GeminiAgent(client, "gemini-3.8-flash")


def catalog_airports(count=20):
    """A spread of scored airports from large to small, chosen by the data rather than by name."""
    scored = [row["airport"] for row in tools._scores()["scored"]]
    return scored[:: max(1, len(scored) // count)][:count]


# The same question, whichever way the model behaves.


def test_normal_tool_use_costs_no_extra_call():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    answer = make_agent(network).ask(QUESTION)
    assert network.modes == ["AUTO", "AUTO"]
    assert answer.text == INDIRECT
    assert [call.name for call in answer.tool_calls] == ["get_airport_profile"]


def test_skipped_tool_call_is_retried_with_a_required_tool_and_the_draft_is_discarded():
    network = FakeNetwork(text_reply(DRAFT), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    answer = make_agent(network).ask(QUESTION)
    assert network.modes == ["AUTO", "ANY", "NONE"]
    assert answer.text == INDIRECT
    assert answer.tool_calls[0].result["profile"]["airport"] == "SFO"
    assert all(DRAFT not in json.dumps(body["contents"]) for body in network.bodies)


def test_both_behaviours_reach_the_same_grounded_outcome():
    normal = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    skipped = FakeNetwork(text_reply(DRAFT), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    first, second = make_agent(normal).ask(QUESTION), make_agent(skipped).ask(QUESTION)
    assert first.text == second.text
    assert first.tool_calls[0].result == second.tool_calls[0].result
    assert normal.bodies[-1]["contents"] == skipped.bodies[-1]["contents"]


def test_retry_leaves_the_tool_choice_to_the_model():
    network = FakeNetwork(text_reply(DRAFT), function_call("compare_airports", {"airports": ["SFO", "OAK"]}), text_reply(INDIRECT))
    answer = make_agent(network).ask(QUESTION)
    config = network.bodies[1]["toolConfig"]["functionCallingConfig"]
    assert config == {"mode": "ANY"}
    assert {item["name"] for item in network.bodies[1]["tools"][0]["functionDeclarations"]} == {
        tool["name"] for tool in tools.TOOL_DEFINITIONS
    }
    assert answer.tool_calls[0].name == "compare_airports"


def test_retrieval_can_still_say_unmet_demand_is_not_measured_while_offering_proxies():
    metrics = tools.get_airport_profile("SFO")["profile"]["demand_pressure_proxies"]["metrics"]
    assert not [name for name in metrics if "unmet" in name]
    growth = metrics["passenger_cagr_3y_pct"]
    reply = f"I can't give an amount for unmet demand. As an indirect sign, SFO's passengers grew {growth}% a year."
    network = FakeNetwork(text_reply("It can't be measured."), function_call("get_airport_profile", PROFILE), text_reply(reply))
    answer = make_agent(network).ask(QUESTION)
    assert answer.text == reply
    assert answer.unverified_numbers == []


def test_a_bare_refusal_with_no_data_still_triggers_retrieval():
    network = FakeNetwork(text_reply("That can't be measured."), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    assert make_agent(network).ask(QUESTION).tool_calls


# When no relevant data arrives, nothing unsupported is shown.


def test_second_skip_withholds_the_answer_and_leaves_no_trace():
    network = FakeNetwork(text_reply(DRAFT), text_reply(DRAFT))
    agent = make_agent(network)
    with pytest.raises(AgentError) as failure:
        agent.ask(QUESTION)
    assert DRAFT not in str(failure.value)
    assert "couldn't retrieve data for SFO" in str(failure.value)
    assert network.modes == ["AUTO", "ANY"]
    assert agent.contents == [] and agent._evidence == [] and agent._retrieved == set()


def test_withheld_answer_does_not_reach_the_next_question():
    network = FakeNetwork(text_reply(DRAFT), text_reply(DRAFT), text_reply("Load factor is passengers divided by seats."))
    agent = make_agent(network)
    with pytest.raises(AgentError):
        agent.ask(QUESTION)
    agent.ask("What does load factor mean?")
    assert DRAFT not in json.dumps(network.bodies[-1]["contents"])
    assert len(network.bodies[-1]["contents"]) == 1


def test_rejected_tool_call_is_not_data():
    network = FakeNetwork(text_reply(DRAFT), profile_call("ZZZZ"), text_reply(DRAFT))
    with pytest.raises(AgentError, match="couldn't retrieve data for SFO"):
        make_agent(network).ask(QUESTION)
    assert network.modes == ["AUTO", "ANY", "NONE"]


def test_data_for_the_wrong_airport_is_retried_then_corrected():
    network = FakeNetwork(profile_call("LAX"), text_reply(DRAFT), profile_call("SFO"), text_reply(INDIRECT))
    answer = make_agent(network).ask(QUESTION)
    assert network.modes == ["AUTO", "AUTO", "ANY", "NONE"]
    assert answer.text == INDIRECT
    assert {call.result["profile"]["airport"] for call in answer.tool_calls} == {"LAX", "SFO"}


def test_data_only_for_the_wrong_airport_is_withheld():
    network = FakeNetwork(profile_call("LAX"), text_reply(DRAFT), profile_call("LAX"), text_reply(DRAFT))
    with pytest.raises(AgentError, match="couldn't retrieve data for SFO"):
        make_agent(network).ask(QUESTION)
    assert len(network.bodies) == 4


def test_unsupported_answer_in_the_last_round_is_withheld_without_another_call():
    network = FakeNetwork(profile_call("LAX"), profile_call("LAX"), text_reply(DRAFT))
    with pytest.raises(AgentError, match="couldn't retrieve data for SFO"):
        make_agent(network).ask(QUESTION)
    assert network.modes == ["AUTO", "AUTO", "NONE"]


def test_every_comparison_airport_named_by_code_must_be_covered():
    network = FakeNetwork(profile_call("LAX"), text_reply("Done."), function_call("compare_airports", {"airports": ["LAX", "SNA"]}), text_reply("Compared."))
    answer = make_agent(network).ask("Compare LAX and SNA.")
    assert network.modes == ["AUTO", "AUTO", "ANY", "NONE"]
    assert answer.text == "Compared."


# The answer can reveal a gap the question did not.


def test_answer_naming_an_airport_with_no_data_is_retried():
    network = FakeNetwork(
        text_reply("ATL is the busiest."),
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        text_reply("Here is what the data shows."),
    )
    assert make_agent(network).ask("Which airport should I study?").tool_calls
    assert network.modes == ["AUTO", "ANY", "NONE"]


def test_invented_figure_with_no_airport_is_retried():
    network = FakeNetwork(
        text_reply("US airports handled 987,654 flights."),
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        text_reply("Here is what the data shows."),
    )
    make_agent(network).ask("How busy are airports?")
    assert network.modes == ["AUTO", "ANY", "NONE"]


def test_invented_figure_with_no_data_withholds_without_naming_an_airport():
    network = FakeNetwork(text_reply("About 987,654 flights."), text_reply("About 987,654 flights."))
    with pytest.raises(AgentError, match="to support this answer"):
        make_agent(network).ask("How busy are airports?")


# General questions stay free of tools.


@pytest.mark.parametrize(
    "question, reply",
    [
        ("What does load factor mean?", "Load factor is passengers divided by seats."),
        ("How is CAGR calculated?", "CAGR is the constant yearly rate linking a start value to an end value."),
        ("What is the Airport Opportunity Score?", "It is a preliminary screening proxy in points out of 100, from 11 weighted metrics."),
        ("What is the difference between 3-year and 10-year growth?", "The 3-year window starts in a recovery year, so it can look stronger."),
        ("Can unmet demand be measured?", "Not directly. Unmet demand would need data on trips people wanted but could not make."),
        ("Hello", "Hello. Ask me about US airports."),
        ("How do airports in the USA compare with airports elsewhere?", "The data covers US airports only."),
    ],
)
def test_general_questions_are_answered_in_one_call_without_a_tool(question, reply):
    network = FakeNetwork(text_reply(reply))
    answer = make_agent(network).ask(question)
    assert answer.text == reply
    assert answer.tool_calls == []
    assert network.modes == ["AUTO"]


def test_clarifying_question_back_to_the_user_is_not_a_claim():
    reply = "Do you mean Portland, Oregon or Portland, Maine?"
    network = FakeNetwork(text_reply(reply))
    assert make_agent(network).ask("What about Portland?").text == reply
    assert network.modes == ["AUTO"]


def test_clarification_cannot_carry_a_claim():
    assert grounding.unsupported("SFO?", "SFO is full. Do you agree?", [], set(), []) == ["SFO"]
    assert grounding.unsupported("SFO?", "Did SFO lose 987,654 flights?", [], set(), []) == ["SFO"]


# Follow-ups can reuse what this conversation already retrieved.


def test_follow_up_reuses_retrieved_data_without_another_tool_call():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply("SFO first."), text_reply("SFO again, for the same reason."))
    agent = make_agent(network)
    agent.ask(QUESTION)
    assert agent.ask("What about SFO's growth?").text == "SFO again, for the same reason."
    assert network.modes == ["AUTO", "AUTO", "AUTO"]


def test_follow_up_about_a_new_airport_needs_new_data():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply("SFO first."), text_reply("LAX is similar."), profile_call("LAX"), text_reply("LAX has data."))
    agent = make_agent(network)
    agent.ask(QUESTION)
    assert agent.ask("What about LAX?").text == "LAX has data."
    assert network.modes == ["AUTO", "AUTO", "AUTO", "ANY", "NONE"]


def test_ranking_rows_count_as_retrieved_data():
    code = tools.rank_opportunity_score([], 1)["ranked"][0]["airport"]
    network = FakeNetwork(function_call("rank_opportunity_score", {"states": [], "top_n": 1}), text_reply("Done."), text_reply(f"{code} leads."))
    agent = make_agent(network)
    agent.ask("Which rank highest?")
    assert agent.ask(f"Why {code}?").text == f"{code} leads."
    assert len(network.bodies) == 3


def test_new_chat_forgets_what_was_retrieved():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply("SFO first."), text_reply(DRAFT), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    agent = make_agent(network)
    agent.ask(QUESTION)
    agent.reset()
    assert agent._retrieved == set()
    agent.ask(QUESTION)
    assert network.modes == ["AUTO", "AUTO", "AUTO", "ANY", "NONE"]


def test_failed_turn_does_not_count_as_retrieved():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), {"promptFeedback": {"blockReason": "SAFETY"}}, text_reply(DRAFT), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    agent = make_agent(network)
    with pytest.raises(AgentError):
        agent.ask(QUESTION)
    assert agent._retrieved == set()
    assert agent.ask(QUESTION).text == INDIRECT


# Airports come from the catalog, so the behaviour holds across all of them.


def test_skip_is_caught_for_airports_of_every_size_by_code_and_by_city():
    checked = 0
    for code in catalog_airports():
        forms = [f"What is the unmet flight demand in {code} airport and why?"]
        forms += [f"What is the unmet flight demand in {name} airport and why?" for name, codes in grounding._catalog()[1].items() if codes == {code}][:1]
        for question in forms:
            network = FakeNetwork(text_reply(DRAFT), profile_call(code), text_reply(INDIRECT))
            answer = make_agent(network).ask(question)
            assert network.modes == ["AUTO", "ANY", "NONE"], question
            assert answer.tool_calls[0].result["profile"]["airport"] == code
            checked += 1
    assert checked >= 30


@pytest.mark.parametrize(
    "text, codes, names",
    [
        (QUESTION, {"SFO"}, set()),
        ("Flights on SFO-LAX", {"SFO", "LAX"}, set()),
        ("Is San Francisco busy?", set(), {"San Francisco"}),
        ("Compare LA and Santa Ana", set(), {"Santa Ana"}),
        ("The Dallas-Fort Worth hub", set(), {"Dallas-Fort Worth"}),
        ("Dallas and Fort Worth", set(), {"Dallas", "Fort Worth"}),
        ("Airports in the USA", set(), set()),
        ("Delta and Southwest carriers", set(), set()),
        ("What does CAGR mean for a hub under the FAA list?", set(), set()),
        ("Which airports in New England look promising?", set(), set()),
    ],
)
def test_airports_are_found_from_the_catalog(text, codes, names):
    assert grounding._coded_airports(text) == codes
    assert set(grounding._named_airports(text)) == names


def test_faa_code_resolves_to_the_code_the_tools_report():
    airports = tools._airports()
    remapped = airports[airports["airport"] != airports["t100_code"]]
    assert len(remapped) > 5
    for row in remapped.itertuples():
        if row.airport not in set(airports["t100_code"]):
            assert grounding._coded_airports(row.airport) == {row.t100_code}


def test_shared_city_is_covered_by_either_of_its_airports():
    codes = grounding._catalog()[1]["Portland"]
    assert len(codes) == 2
    for code in codes:
        assert grounding.unsupported("About Portland", "Portland looks fine.", [], {code}, []) is None


def test_city_that_is_also_a_state_does_not_block_an_answer_that_used_a_tool():
    arguments = {"window_years": 3, "states": ["NY"], "top_n": 1, "min_enplanements": 100000}
    network = FakeNetwork(function_call("rank_airports", arguments), text_reply("Here is the ranking."))
    assert make_agent(network).ask("Rank New York airports by growth.").text == "Here is the ranking."
    assert network.modes == ["AUTO", "AUTO"]


def test_answer_may_mention_another_airport_once_it_has_data():
    network = FakeNetwork(function_call("get_airport_profile", PROFILE), text_reply("SFO grew, unlike Fort Worth in other years."))
    assert make_agent(network).ask(QUESTION).tool_calls
    assert network.modes == ["AUTO", "AUTO"]


def test_lower_case_code_is_caught_when_the_answer_names_the_airport():
    network = FakeNetwork(text_reply("SFO is very busy."), function_call("get_airport_profile", PROFILE), text_reply(INDIRECT))
    assert make_agent(network).ask("how busy is sfo?").tool_calls


def test_retrieved_airports_ignore_failures_and_notes():
    good = ToolCall("get_airport_profile", {}, tools.get_airport_profile("SFO"), False)
    failed = ToolCall("get_airport_profile", {}, {"airport": "LAX", "error": "Bad"}, True)
    note = ToolCall("x", {}, {"note": "Unlike LAX or ANC, this is only a note."}, False)
    assert grounding.retrieved_airports([good, failed, note]) == {"SFO"}


def test_a_turn_makes_at_most_one_more_call_than_the_tool_round_limit():
    network = FakeNetwork(profile_call("LAX"), text_reply(DRAFT), profile_call("LAX"), text_reply(DRAFT))
    with pytest.raises(AgentError):
        make_agent(network).ask(QUESTION)
    assert len(network.bodies) == gemini_agent.MAX_TOOL_ROUNDS + 1
