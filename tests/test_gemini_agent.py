"""Gemini agent checks using the real SDK over a fake network.

No request leaves this machine and no API allowance is used.
"""

import base64
import json
import re

import httpx
import pytest
from google import genai
from google.genai import errors, types

from src import config, gemini_agent, tools
from src.agent import AgentError
from src.gemini_agent import GeminiAgent, select_model

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "ontime_airport_month.parquet").exists(),
    reason="Processed data not built",
)

SIGNATURE = base64.b64encode(b"signature").decode()


def function_call(name, args, call_id="call_1"):
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"id": call_id, "name": name, "args": args}, "thoughtSignature": SIGNATURE}],
            },
            "finishReason": "STOP",
        }]
    }


def text_reply(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}


class FakeNetwork:
    def __init__(self, *replies, status=200):
        self.replies = list(replies)
        self.status = status
        self.bodies = []
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-embedding", "supportedGenerationMethods": ["embedContent"]},
            ]})
        self.bodies.append(json.loads(request.content))
        return httpx.Response(self.status, json=self.replies.pop(0))


def make_client(network):
    http_options = types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(network)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key="test-key", http_options=http_options)


def make_agent(network, **kwargs):
    return GeminiAgent(make_client(network), "gemini-3.8-flash", **kwargs)


def test_function_call_runs_real_python_tool_and_returns_the_result():
    network = FakeNetwork(
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}),
        text_reply("Scheduled passenger flights: 2.6% long-haul."),
    )
    answer = make_agent(network).ask("What share of ANC flights are long-haul?")

    assert answer.text == "Scheduled passenger flights: 2.6% long-haul."
    assert answer.unverified_numbers == []
    assert answer.tool_calls[0].result["scopes"]["scheduled_passenger"]["share_pct"] == 2.6
    returned = network.bodies[1]["contents"][-1]
    assert returned["role"] == "user"
    response = returned["parts"][0]["functionResponse"]
    assert response["id"] == "call_1"
    assert response["response"]["scopes"]["scheduled_passenger"]["share_pct"] == 2.6


def test_request_declares_every_tool_and_sets_automatic_mode():
    network = FakeNetwork(text_reply("Hello."))
    make_agent(network).ask("Hello")
    body = network.bodies[0]
    names = {item["name"] for item in body["tools"][0]["functionDeclarations"]}
    assert names == {tool["name"] for tool in tools.TOOL_DEFINITIONS}
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    instruction = body["systemInstruction"]["parts"][0]["text"]
    assert "preliminary screening proxy" in instruction
    assert "Never calculate, approximate, adjust, rescale or reweight a score" in instruction
    assert "never a probability" in instruction
    assert "part of the score uses cargo-inclusive departure metrics" in instruction
    assert "absolute passenger growth favors larger airports" in instruction
    assert "Never call it a passenger load factor" in instruction
    assert "80 to 120 words" in instruction
    assert "Never write a table" in instruction
    assert "Never add a passenger threshold" in instruction
    assert "terminal capacity" in instruction
    assert body["generationConfig"]["maxOutputTokens"] > 0
    declarations = {item["name"]: item for item in body["tools"][0]["functionDeclarations"]}
    rank_by = declarations["rank_airports"]["parameters_json_schema"]["properties"]["rank_by"]
    assert rank_by["enum"] == ["absolute_passenger_growth", "passenger_cagr"]


def test_integer_enums_move_to_the_description_and_tools_stay_unchanged():
    before = json.dumps(tools.TOOL_DEFINITIONS, sort_keys=True)
    schemas = {tool.name: tool.parameters_json_schema for tool in gemini_agent.tool_declarations()[0].function_declarations}
    window = schemas["rank_airports"]["properties"]["window_years"]
    assert "enum" not in window
    assert "Allowed values: 3, 10" in window["description"]
    assert "additionalProperties" not in schemas["rank_airports"]
    assert json.dumps(tools.TOOL_DEFINITIONS, sort_keys=True) == before


def test_score_comes_from_the_python_tool_and_is_returned_to_the_model():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1.0}),
        text_reply("ILG scored 78.38."),
    )
    answer = make_agent(network).ask("Which airport has the highest score?")
    assert answer.unverified_numbers == []
    assert answer.tool_calls[0].result["ranked"][0]["airport"] == "ILG"
    returned = network.bodies[1]["contents"][-1]["parts"][0]["functionResponse"]["response"]
    assert returned["ranked"][0]["opportunity_score"] == 78.38
    assert returned["disclaimer"].startswith("Preliminary screening proxy")


@pytest.mark.parametrize(
    "question, first_reply",
    [
        ("Which airports should we investigate for terminal expansion?", text_reply("Answer.")),
        ("What is the load factor at SFO?", function_call("get_airport_profile", {"airport": "SFO"})),
        ("Compare LAX and SNA.", function_call("compare_airports", {"airports": ["LAX", "SNA"]})),
    ],
)
def test_every_question_is_offered_all_tools_in_automatic_mode(question, first_reply):
    network = FakeNetwork(first_reply, text_reply("Answer."))
    make_agent(network).ask(question)
    body = network.bodies[0]
    assert {item["name"] for item in body["tools"][0]["functionDeclarations"]} == {
        tool["name"] for tool in tools.TOOL_DEFINITIONS
    }
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert "allowedFunctionNames" not in body["toolConfig"]["functionCallingConfig"]


def test_model_can_combine_tools_in_one_turn():
    network = FakeNetwork(
        {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"id": "a", "name": "rank_opportunity_score", "args": {"states": [], "top_n": 3}}},
            {"functionCall": {"id": "b", "name": "compare_airports", "args": {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}}},
        ]}, "finishReason": "STOP"}]},
        text_reply("Combined answer."),
    )
    answer = make_agent(network).ask("Growth and congestion?")
    assert [call.name for call in answer.tool_calls] == ["rank_opportunity_score", "compare_airports"]
    assert len(network.bodies[1]["contents"][-1]["parts"]) == 2


def test_model_can_use_tools_across_rounds():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1}),
        function_call("get_airport_profile", {"airport": "ILG", "table_sections": []}),
        text_reply("Done."),
    )
    answer = make_agent(network).ask("Top airport, then its detail?")
    assert [call.name for call in answer.tool_calls] == ["rank_opportunity_score", "get_airport_profile"]


def test_follow_up_sends_earlier_constraints_to_the_model():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3, "min_passengers_2025": 2000000, "passenger_comparison": "over"}),
        text_reply("First."),
        text_reply("Second."),
    )
    agent = make_agent(network)
    agent.ask("Top airports with over 2 million passengers?")
    agent.ask("And what about in Texas?")
    earlier = json.dumps(network.bodies[2]["contents"])
    assert "over 2 million passengers" in earlier
    assert '"min_passengers_2025": 2000000' in earlier


def test_whole_number_float_threshold_from_the_model_is_accepted():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 2.0, "min_passengers_2025": 2000000.0, "passenger_comparison": "over"}),
        text_reply("Done."),
    )
    call = make_agent(network).ask("Over 2 million?").tool_calls[0]
    assert not call.is_error
    assert call.result["filters"]["passengers_2025"] == "Over 2,000,000 2025 departing passengers"


def test_thought_signature_is_sent_back_unchanged():
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}),
        text_reply("Done."),
    )
    make_agent(network).ask("Tell me about SFO")
    model_turn = network.bodies[1]["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["thoughtSignature"] == SIGNATURE


def test_whole_number_floats_become_integers():
    network = FakeNetwork(
        function_call("rank_airports", {"window_years": 3.0, "states": [], "top_n": 2.0, "min_enplanements": 100000.0}),
        text_reply("Done."),
    )
    call = make_agent(network).ask("Top two").tool_calls[0]
    assert not call.is_error
    assert len(call.result["ranked"]) == 2


def test_invented_number_is_flagged():
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}),
        text_reply("SFO grew 8.87% a year and has 1,234,567 unmet passengers."),
    )
    assert make_agent(network).ask("Tell me about SFO").unverified_numbers == ["1,234,567"]


def test_bad_tool_input_returns_an_error_the_model_can_correct():
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ZZZZ"}),
        text_reply("I could not find that airport."),
    )
    answer = make_agent(network).ask("Tell me about ZZZZ")
    assert answer.tool_calls[0].is_error
    assert "error" in network.bodies[1]["contents"][-1]["parts"][0]["functionResponse"]["response"]


def test_follow_up_keeps_history():
    network = FakeNetwork(text_reply("First."), text_reply("Second."))
    agent = make_agent(network)
    agent.ask("One")
    agent.ask("Two")
    assert [item["role"] for item in network.bodies[1]["contents"]] == ["user", "model", "user"]


def test_endless_tool_calls_stop_at_the_round_limit():
    replies = [function_call("get_airport_profile", {"airport": "SFO"}) for _ in range(gemini_agent.MAX_TOOL_ROUNDS)]
    network = FakeNetwork(*replies)
    with pytest.raises(AgentError, match="tool rounds"):
        make_agent(network).ask("Question")
    assert len(network.bodies) == gemini_agent.MAX_TOOL_ROUNDS


def calling_modes(network):
    return [body["toolConfig"]["functionCallingConfig"]["mode"] for body in network.bodies]


def test_third_call_uses_mode_none_after_two_automatic_calls():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1}),
        function_call("get_airport_profile", {"airport": "ILG", "table_sections": []}),
        text_reply("Done."),
    )
    answer = make_agent(network).ask("Top airport, then its detail?")
    assert answer.text == "Done."
    assert calling_modes(network) == ["AUTO", "AUTO", "NONE"]
    # Earlier calls and responses stay valid only while the declarations are still sent.
    final_request = network.bodies[2]
    assert {item["name"] for item in final_request["tools"][0]["functionDeclarations"]} == {
        tool["name"] for tool in tools.TOOL_DEFINITIONS
    }
    assert final_request["systemInstruction"] == network.bodies[0]["systemInstruction"]
    assert final_request["generationConfig"] == network.bodies[0]["generationConfig"]


def test_answer_before_the_final_call_never_uses_mode_none():
    network = FakeNetwork(function_call("get_airport_profile", {"airport": "SFO"}), text_reply("Done."))
    make_agent(network).ask("Tell me about SFO")
    assert calling_modes(network) == ["AUTO", "AUTO"]


def test_each_question_gets_its_own_three_calls():
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}),
        function_call("get_airport_profile", {"airport": "LAX"}),
        text_reply("First."),
        function_call("get_airport_profile", {"airport": "SNA"}),
        text_reply("Second."),
    )
    agent = make_agent(network)
    agent.ask("One")
    agent.ask("Two")
    assert calling_modes(network) == ["AUTO", "AUTO", "NONE", "AUTO", "AUTO"]


def test_results_from_every_round_reach_the_final_call_and_the_answer():
    network = FakeNetwork(
        {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"id": "a", "name": "rank_opportunity_score", "args": {"states": [], "top_n": 1}},
             "thoughtSignature": SIGNATURE},
            {"functionCall": {"id": "b", "name": "get_long_haul_share", "args": {"airport": "ANC", "threshold_miles": 3000}}},
        ]}, "finishReason": "STOP"}]},
        function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}, call_id="c"),
        text_reply("ILG scored 78.38, ANC is 2.6% long-haul, SFO grew 8.87% a year, and 1,234,567 is invented."),
    )
    agent = make_agent(network)
    answer = agent.ask("Combine these")

    assert [call.name for call in answer.tool_calls] == [
        "rank_opportunity_score", "get_long_haul_share", "get_airport_profile",
    ]
    assert not any(call.is_error for call in answer.tool_calls)

    final_contents = network.bodies[2]["contents"]
    assert [item["role"] for item in final_contents] == ["user", "model", "user", "model", "user"]
    assert final_contents[1]["parts"][0]["thoughtSignature"] == SIGNATURE
    responses = {
        part["functionResponse"]["id"]: part["functionResponse"]["response"]
        for item in final_contents
        for part in item["parts"]
        if "functionResponse" in part
    }
    assert set(responses) == {"a", "b", "c"}
    assert responses["a"]["ranked"][0]["opportunity_score"] == 78.38
    assert responses["b"]["scopes"]["scheduled_passenger"]["share_pct"] == 2.6
    assert responses["c"]["profile"]["demand_pressure_proxies"]["metrics"]["passenger_cagr_3y_pct"] == 8.87

    # Each number is checked against the evidence from whichever round produced it.
    assert answer.unverified_numbers == ["1,234,567"]
    assert len(agent._evidence) == 4


@pytest.mark.parametrize(
    "final_parts",
    [
        [{"functionCall": {"id": "z", "name": "get_airport_profile", "args": {"airport": "SFO"}}}],
        [
            {"text": "An answer that also asks for a tool."},
            {"functionCall": {"id": "z", "name": "get_airport_profile", "args": {"airport": "SFO"}}},
        ],
    ],
    ids=["tool_call_only", "text_and_tool_call"],
)
def test_unexpected_final_tool_call_runs_nothing_and_leaves_history_clean(monkeypatch, final_parts):
    executed = []
    real_execute = tools.execute_tool

    def spy(name, tool_input):
        executed.append(name)
        return real_execute(name, tool_input)

    monkeypatch.setattr(tools, "execute_tool", spy)
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1}),
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}),
        {"candidates": [{"content": {"role": "model", "parts": final_parts}, "finishReason": "STOP"}]},
        text_reply("Next answer."),
    )
    agent = make_agent(network)

    with pytest.raises(AgentError, match="tool rounds"):
        agent.ask("Question")

    assert executed == ["rank_opportunity_score", "get_long_haul_share"]
    assert calling_modes(network) == ["AUTO", "AUTO", "NONE"]
    assert len(network.bodies) == 3
    assert agent.contents == []
    assert agent._evidence == []

    assert agent.ask("Next question").text == "Next answer."
    assert [item["role"] for item in network.bodies[3]["contents"]] == ["user"]
    assert calling_modes(network)[3] == "AUTO"


def test_empty_final_reply_is_reported_and_leaves_history_clean():
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}),
        function_call("get_airport_profile", {"airport": "LAX"}),
        {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "MALFORMED_FUNCTION_CALL"}]},
    )
    agent = make_agent(network)
    with pytest.raises(AgentError, match="no text"):
        agent.ask("Question")
    assert agent.contents == []
    assert agent._evidence == []


def user_text(body):
    return body["contents"][-1]["parts"][0]["text"]


def shortlist(count=5):
    return [row["airport"] for row in tools.rank_opportunity_score([], count)["ranked"]]


def note_for(airports):
    return f"(App note: the main set in the previous answer was {', '.join(airports)}.)"


def test_follow_up_receives_the_exact_main_set_not_a_supporting_example():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 5}),
        function_call("rank_airports", {"window_years": 3, "states": [], "top_n": 25, "min_enplanements": 100000}),
        text_reply("Start with this shortlist. Another airport is a supporting example."),
        text_reply("Which group do you mean?"),
    )
    agent = make_agent(network)
    first = agent.ask("Where would you start?")
    agent.ask("Which of those grew the most?")

    supporting = {row["airport"] for row in first.tool_calls[1].result["ranked"]} - set(shortlist())
    assert supporting
    assert "App note" not in user_text(network.bodies[0])
    sent = user_text(network.bodies[3])
    assert sent.startswith("Which of those grew the most?")
    assert note_for(shortlist()) in sent
    assert not any(code in sent for code in supporting)


def test_follow_up_note_leaves_tool_choice_and_evidence_alone():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 2}), text_reply("First."), text_reply("Second."),
    )
    agent = make_agent(network)
    agent.ask("One")
    agent.ask("Two")
    body = network.bodies[2]
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert "allowedFunctionNames" not in body["toolConfig"]["functionCallingConfig"]
    assert {item["name"] for item in body["tools"][0]["functionDeclarations"]} == {
        tool["name"] for tool in tools.TOOL_DEFINITIONS
    }
    assert "Two" in agent._evidence
    assert not any("App note" in item for item in agent._evidence)


def test_clarifying_answer_without_a_tool_keeps_the_set():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        text_reply("Do you mean the shortlist?"),
        text_reply("Done."),
    )
    agent = make_agent(network)
    agent.ask("Start?")
    agent.ask("Which one?")
    agent.ask("The shortlist.")
    assert note_for(shortlist(3)) in user_text(network.bodies[3])


def test_a_new_comparison_replaces_the_set_and_earlier_notes_stay_in_history():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["passenger_growth"]}), text_reply("Compared."),
        text_reply("Done."),
    )
    agent = make_agent(network)
    agent.ask("Start?")
    agent.ask("What about LAX and SNA?")
    agent.ask("Which of them?")
    last = network.bodies[4]
    assert note_for(["LAX", "SNA"]) in user_text(last)
    assert note_for(shortlist(3)) in json.dumps(last["contents"])


def test_failed_turn_does_not_change_the_set():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": []}),
        {"promptFeedback": {"blockReason": "SAFETY"}},
        text_reply("Done."),
    )
    agent = make_agent(network)
    agent.ask("Start?")
    with pytest.raises(AgentError):
        agent.ask("What about LAX and SNA?")
    agent.ask("Which of them?")
    assert note_for(shortlist(3)) in user_text(network.bodies[4])


def test_rejected_tool_call_does_not_change_the_set():
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        function_call("get_airport_profile", {"airport": "ZZZZ", "table_sections": []}), text_reply("Not found."),
        text_reply("Done."),
    )
    agent = make_agent(network)
    agent.ask("Start?")
    agent.ask("ZZZZ?")
    agent.ask("Which of them?")
    assert note_for(shortlist(3)) in user_text(network.bodies[4])


def test_system_prompt_covers_follow_ups_tone_and_length():
    instruction = make_agent(FakeNetwork())._config.system_instruction
    assert "Never add, drop or swap an airport" in instruction
    assert "supporting example" in instruction
    assert "one short clarifying question" in instruction
    assert "experienced analyst thinking out loud" in instruction
    assert "No hype" in instruction
    assert "what you would check next" in instruction
    assert "40 to 80 words" in instruction
    assert "passenger_growth when only passenger traffic or growth matters" in instruction


def test_passenger_growth_section_is_offered_to_the_model():
    declarations = gemini_agent.tool_declarations()[0].function_declarations
    schemas = {tool.name: tool.parameters_json_schema for tool in declarations}
    for name in ("compare_airports", "get_airport_profile"):
        assert "passenger_growth" in schemas[name]["properties"]["table_sections"]["items"]["enum"]


def test_main_table_line_is_removed_and_names_the_ranking_and_next_set():
    growth_args = {"window_years": 3, "states": [], "top_n": 3, "min_enplanements": 100000}
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        function_call("rank_airports", growth_args),
        text_reply("Growth answer.\n\n[primary_table: rank_airports]"),
        text_reply("Done."),
    )
    agent = make_agent(network)
    answer = agent.ask("How fast?")
    agent.ask("Which of them?")
    assert answer.text == "Growth answer."
    assert answer.primary_table == "rank_airports"
    assert answer.unverified_numbers == []
    expected = [row["airport"] for row in answer.tool_calls[1].result["ranked"]]
    assert note_for(expected) in user_text(network.bodies[3])


def test_reply_with_only_a_main_table_line_is_reported_as_no_text():
    network = FakeNetwork(text_reply("[primary_table: rank_airports]"))
    with pytest.raises(AgentError, match="no text"):
        make_agent(network).ask("Question")


def test_agent_has_no_session_call_limit():
    questions = 12
    network = FakeNetwork(*[text_reply("Answer.") for _ in range(questions)])
    agent = make_agent(network)
    for position in range(questions):
        assert agent.ask(f"Question {position}").text == "Answer."
    assert len(network.bodies) == questions
    source = (config.ROOT / "src" / "gemini_agent.py").read_text()
    assert "call_limit" not in source and "calls_used" not in source and "GEMINI_MAX_CALLS" not in source


def test_per_question_tool_rounds_are_still_limited():
    network = FakeNetwork(*[function_call("get_airport_profile", {"airport": "SFO"}) for _ in range(gemini_agent.MAX_TOOL_ROUNDS)])
    with pytest.raises(AgentError, match="tool rounds"):
        make_agent(network).ask("Question")
    assert len(network.bodies) == gemini_agent.MAX_TOOL_ROUNDS


def test_usable_models_lists_only_preferred_models_the_key_can_use():
    network = FakeNetwork()
    assert gemini_agent.usable_models(make_client(network)) == ["gemini-3.6-flash", "gemini-3.8-flash"]
    assert network.bodies == []


def test_usable_models_can_add_one_named_model_only_if_available():
    client = make_client(FakeNetwork())
    assert gemini_agent.usable_models(client, "gemini-9-flash") == ["gemini-3.6-flash", "gemini-3.8-flash"]


def test_usable_models_explains_when_none_are_available():
    def embeddings_only(request):
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-embedding", "supportedGenerationMethods": ["embedContent"]},
        ]})

    with pytest.raises(AgentError, match="isn't available right now"):
        gemini_agent.usable_models(make_client(embeddings_only))


def listed_models(*names):
    return lambda request: httpx.Response(200, json={"models": [
        {"name": f"models/{name}", "supportedGenerationMethods": ["generateContent"]} for name in names
    ]})


LITE_35, LITE_31, FLASH_36, FLASH_38 = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"


def test_model_priority_order():
    assert gemini_agent.PREFERRED_MODELS == (LITE_35, LITE_31, FLASH_36, FLASH_38)


@pytest.mark.parametrize(
    "available, requested, expected",
    [
        ((FLASH_38, FLASH_36, LITE_31, LITE_35), None, (LITE_35, LITE_31)),
        ((FLASH_38, FLASH_36, LITE_31), None, (LITE_31, FLASH_36)),
        ((FLASH_38, FLASH_36, LITE_35), None, (LITE_35, FLASH_36)),
        ((FLASH_38, FLASH_36), None, (FLASH_36, FLASH_38)),
        ((FLASH_38,), None, (FLASH_38, None)),
        ((FLASH_38, FLASH_36, LITE_31, LITE_35), FLASH_36, (FLASH_36, LITE_35)),
        ((FLASH_38, FLASH_36, LITE_31, LITE_35), "gemini-9-flash", (LITE_35, LITE_31)),
        (("gemini-9-flash", FLASH_36), "gemini-9-flash", ("gemini-9-flash", FLASH_36)),
    ],
)
def test_model_pair_picks_one_model_and_at_most_one_fallback(available, requested, expected):
    assert gemini_agent.model_pair(make_client(listed_models(*available)), requested) == expected


class StatusNetwork:
    """Answer each generate request from a list of (status, body) pairs and record the model it named."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.models = []
        self.bodies = []

    def __call__(self, request):
        self.models.append(request.url.path.split("/models/")[1].split(":")[0])
        self.bodies.append(json.loads(request.content))
        status, body = self.replies.pop(0)
        return httpx.Response(status, json=body)


def api_error(code, status, message="RAW GOOGLE TEXT", *violations):
    details = [{"violations": list(violations)}] if violations else []
    return code, {"error": {"code": code, "status": status, "message": message, "details": details}}


PER_MODEL = {"quotaId": "GenerateRequestsPerDayPerProjectPerModel", "quotaDimensions": {"model": "gemini-3.8-flash"}}
KEY_WIDE = {"quotaId": "GenerateRequestsPerDayPerProject"}
MODEL_ERRORS = {
    "not_found": api_error(404, "NOT_FOUND"),
    "per_model_quota": api_error(429, "RESOURCE_EXHAUSTED", "quota", PER_MODEL),
    "overloaded_model": api_error(503, "UNAVAILABLE", "This model is overloaded."),
}
OTHER_ERRORS = {
    "unauthenticated": api_error(401, "UNAUTHENTICATED"),
    "forbidden": api_error(403, "PERMISSION_DENIED", "This model is not allowed."),
    "invalid_key": api_error(400, "INVALID_ARGUMENT", "API_KEY_INVALID"),
    "bad_request": api_error(400, "INVALID_ARGUMENT", "Unsupported model input."),
    "key_wide_quota": api_error(429, "RESOURCE_EXHAUSTED", "quota", KEY_WIDE),
    "mixed_quota": api_error(429, "RESOURCE_EXHAUSTED", "quota", PER_MODEL, KEY_WIDE),
    "unspecified_quota": api_error(429, "RESOURCE_EXHAUSTED", "model quota"),
    "server_error": api_error(500, "INTERNAL", "The model failed."),
    "general_outage": api_error(503, "UNAVAILABLE", "The service is unavailable."),
}


def fallback_agent(network):
    return GeminiAgent(make_client(network), "gemini-3.8-flash", "gemini-3.6-flash")


@pytest.mark.parametrize("failure", MODEL_ERRORS.values(), ids=MODEL_ERRORS)
def test_model_specific_error_moves_to_the_fallback_and_stays_there(failure):
    network = StatusNetwork(failure, (200, text_reply("First.")), (200, text_reply("Second.")))
    agent = fallback_agent(network)
    assert agent.ask("One?").text == "First."
    assert agent.ask("Two?").text == "Second."
    assert network.models == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.6-flash"]
    # Both models get the same request, including the tools.
    assert network.bodies[0] == network.bodies[1]
    assert network.bodies[0]["tools"]


@pytest.mark.parametrize("failure", OTHER_ERRORS.values(), ids=OTHER_ERRORS)
def test_other_errors_are_not_retried(failure):
    network = StatusNetwork(failure)
    agent = fallback_agent(network)
    with pytest.raises(errors.APIError):
        agent.ask("Question")
    assert network.models == ["gemini-3.8-flash"]
    assert agent.model == "gemini-3.8-flash"
    assert agent.contents == []


def test_timeout_is_not_retried_on_the_fallback():
    requests = []

    def stalled(request):
        requests.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(AgentError, match="did not respond in time"):
        fallback_agent(stalled).ask("Question")
    assert len(requests) == 1


def test_failure_after_a_tool_call_resends_the_results_without_running_the_tool_again(monkeypatch):
    executed = []
    real_tool = tools.execute_tool

    def counted(name, arguments):
        executed.append(name)
        return real_tool(name, arguments)

    monkeypatch.setattr(tools, "execute_tool", counted)
    network = StatusNetwork(
        (200, function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000})),
        MODEL_ERRORS["per_model_quota"],
        (200, text_reply("Scheduled passenger flights: 2.6% long-haul.")),
    )
    answer = fallback_agent(network).ask("What share of ANC flights are long-haul?")

    assert executed == ["get_long_haul_share"]
    assert len(answer.tool_calls) == 1
    assert answer.unverified_numbers == []
    assert network.models == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.6-flash"]
    # The fallback gets the identical history, with the thought signature and the tool result.
    assert network.bodies[2] == network.bodies[1]
    model_turn, tool_turn = network.bodies[2]["contents"][1:3]
    assert model_turn["parts"][0]["thoughtSignature"] == SIGNATURE
    assert "functionResponse" in tool_turn["parts"][0]


def test_both_models_unavailable_gives_one_simple_message_after_two_requests():
    network = StatusNetwork(MODEL_ERRORS["per_model_quota"], MODEL_ERRORS["not_found"])
    agent = fallback_agent(network)
    with pytest.raises(AgentError) as caught:
        agent.ask("Question")
    assert str(caught.value) == gemini_agent.MODELS_UNAVAILABLE_MESSAGE
    assert "model" not in str(caught.value).lower()
    assert len(network.models) == 2
    assert agent.contents == []


def test_fallback_failing_for_another_reason_keeps_that_error():
    network = StatusNetwork(MODEL_ERRORS["not_found"], OTHER_ERRORS["key_wide_quota"])
    with pytest.raises(errors.ClientError):
        fallback_agent(network).ask("Question")
    assert len(network.models) == 2


def test_fallback_is_used_at_most_once_per_conversation():
    network = StatusNetwork(MODEL_ERRORS["not_found"], (200, text_reply("First.")), MODEL_ERRORS["not_found"])
    agent = fallback_agent(network)
    agent.ask("One?")
    with pytest.raises(errors.ClientError):
        agent.ask("Two?")
    assert network.models == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.6-flash"]


def test_reset_returns_to_the_preferred_model_with_its_fallback():
    network = StatusNetwork(
        MODEL_ERRORS["not_found"], (200, text_reply("First.")), MODEL_ERRORS["not_found"], (200, text_reply("Fresh.")),
    )
    agent = fallback_agent(network)
    agent.ask("One?")
    agent.reset()
    assert agent.model == "gemini-3.8-flash"
    assert agent.ask("Two?").text == "Fresh."
    assert network.models[2:] == ["gemini-3.8-flash", "gemini-3.6-flash"]


def test_grounding_still_applies_after_a_fallback():
    network = StatusNetwork(
        MODEL_ERRORS["not_found"],
        (200, text_reply("ANC has 1,234,567 unmet passengers.")),
        (200, function_call("get_airport_profile", {"airport": "ANC"})),
        (200, text_reply("ANC profile retrieved.")),
    )
    answer = fallback_agent(network).ask("Unmet demand at ANC?")
    assert answer.text == "ANC profile retrieved."
    assert [call.name for call in answer.tool_calls] == ["get_airport_profile"]
    assert set(network.models[1:]) == {"gemini-3.6-flash"}


def test_error_without_a_fallback_is_raised_unchanged():
    network = StatusNetwork(MODEL_ERRORS["not_found"])
    with pytest.raises(errors.ClientError):
        GeminiAgent(make_client(network), "gemini-3.8-flash").ask("Question")
    assert len(network.models) == 1


def test_quota_error_makes_one_call_and_leaves_history_clean():
    network = FakeNetwork({"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}, status=429)
    agent = make_agent(network)
    with pytest.raises(errors.ClientError):
        agent.ask("Question")
    assert len(network.bodies) == 1
    assert agent.contents == []


def test_build_client_disables_retries():
    client = gemini_agent.build_client("test-key")
    assert client._api_client._http_options.retry_options.attempts == 1


def test_build_client_sets_a_60_second_timeout():
    client = gemini_agent.build_client("test-key")
    assert client._api_client._http_options.timeout == 60_000


def test_timeout_makes_one_call_and_leaves_history_clean():
    requests = []

    def stalled(request):
        requests.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    agent = make_agent(stalled)
    with pytest.raises(AgentError, match="did not respond in time"):
        agent.ask("Question")
    assert len(requests) == 1
    assert agent.contents == []


def test_timeout_while_listing_models_is_reported():
    def stalled(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(AgentError, match="did not respond in time"):
        select_model(make_client(stalled))


def test_blocked_prompt_is_reported():
    network = FakeNetwork({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(AgentError, match="no answer"):
        make_agent(network).ask("Question")


def test_empty_reply_is_reported_not_returned_blank():
    network = FakeNetwork({"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(AgentError, match="no text"):
        make_agent(network).ask("Question")


def test_thought_text_is_not_shown_as_the_answer():
    network = FakeNetwork({"candidates": [{"content": {"role": "model", "parts": [
        {"text": "private reasoning", "thought": True}, {"text": "Visible answer."},
    ]}, "finishReason": "STOP"}]})
    assert make_agent(network).ask("Question").text == "Visible answer."


def test_model_selection_follows_the_priority_order():
    assert select_model(make_client(FakeNetwork())) == "gemini-3.6-flash"
    assert select_model(make_client(listed_models("gemini-3.8-flash", "gemini-3.1-flash-lite"))) == "gemini-3.1-flash-lite"


def test_requested_model_must_be_available():
    client = make_client(FakeNetwork())
    assert select_model(client, "gemini-3.6-flash") == "gemini-3.6-flash"
    with pytest.raises(AgentError, match="not available"):
        select_model(client, "gemini-9-flash")


def test_listing_models_generates_no_content():
    network = FakeNetwork()
    select_model(make_client(network))
    assert network.bodies == []
    assert all(request.method == "GET" for request in network.requests)


NO_TOKENS = " prompt_tokens=None thinking_tokens=None output_tokens=None"


def timing_events(caplog):
    return [record.getMessage() for record in caplog.records if record.name == gemini_agent.TIMING_LOGGER.name]


def timed_ask(caplog, network, question="Question?"):
    with caplog.at_level("INFO", logger=gemini_agent.TIMING_LOGGER.name):
        try:
            fallback_agent(network).ask(question)
        except (errors.APIError, AgentError):
            pass
    return timing_events(caplog)


def test_timing_covers_each_request_and_tool_and_the_question_total(caplog):
    network = StatusNetwork(
        (200, function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000})),
        (200, text_reply("Scheduled passenger flights: 2.6% long-haul.")),
    )
    events = timed_ask(caplog, network, "What share of ANC flights are long-haul?")
    assert [event.split()[0] for event in events] == ["gemini_request", "tool", "gemini_request", "question_total"]
    assert re.fullmatch(r"gemini_request model=gemini-3\.8-flash status=ok duration_ms=\d+" + NO_TOKENS, events[0])
    assert re.fullmatch(r"tool name=get_long_haul_share is_error=False duration_ms=\d+", events[1])
    assert re.fullmatch(
        r"question_total duration_ms=\d+ requests=2 request_ms=\d+ tools=1 tool_ms=\d+ "
        r"fallback=no grounding_retry=no other_ms=\d+ outcome=answered",
        events[3],
    )


def test_timing_records_a_fallback_and_both_models(caplog):
    events = timed_ask(caplog, StatusNetwork(MODEL_ERRORS["not_found"], (200, text_reply("Answered."))))
    assert re.fullmatch(r"gemini_request model=gemini-3\.8-flash status=404 duration_ms=\d+" + NO_TOKENS, events[0])
    assert events[1] == "gemini_fallback from_model=gemini-3.8-flash to_model=gemini-3.6-flash status=404"
    assert re.fullmatch(r"gemini_request model=gemini-3\.6-flash status=ok duration_ms=\d+" + NO_TOKENS, events[2])
    assert "requests=2" in events[3] and "fallback=yes" in events[3]


def test_timing_records_a_grounding_retry(caplog):
    network = StatusNetwork(
        (200, text_reply("ANC has 1,234,567 unmet passengers.")),
        (200, function_call("get_airport_profile", {"airport": "ANC"})),
        (200, text_reply("ANC profile retrieved.")),
    )
    events = timed_ask(caplog, network, "Unmet demand at ANC?")
    assert "grounding_retry round=1" in events
    assert "requests=3" in events[-1] and "grounding_retry=yes" in events[-1] and "fallback=no" in events[-1]


def test_timing_records_failures_by_type_only(caplog):
    events = timed_ask(caplog, StatusNetwork(OTHER_ERRORS["unauthenticated"]))
    assert re.fullmatch(r"gemini_request model=gemini-3\.8-flash status=401 duration_ms=\d+" + NO_TOKENS, events[0])
    assert "fallback=no" in events[1] and events[1].endswith("outcome=ClientError")


def test_timing_records_a_timeout(caplog):
    def stalled(request):
        raise httpx.ReadTimeout("timed out", request=request)

    events = timed_ask(caplog, stalled)
    assert re.fullmatch(r"gemini_request model=gemini-3\.8-flash status=no_response duration_ms=\d+" + NO_TOKENS, events[0])
    assert events[1].endswith("outcome=AgentError")


def test_timing_records_token_counts_when_gemini_reports_them(caplog):
    reply = text_reply("Answered.")
    reply["usageMetadata"] = {"promptTokenCount": 1200, "thoughtsTokenCount": 3400, "candidatesTokenCount": 56}
    events = timed_ask(caplog, StatusNetwork((200, reply)))
    assert re.fullmatch(
        r"gemini_request model=gemini-3\.8-flash status=ok duration_ms=\d+ "
        r"prompt_tokens=1200 thinking_tokens=3400 output_tokens=56",
        events[0],
    )


def test_timing_never_logs_keys_questions_answers_tool_data_or_google_text(caplog):
    network = StatusNetwork(
        (200, function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000})),
        api_error(429, "RESOURCE_EXHAUSTED", "GOOGLETEXT", PER_MODEL),
        (200, text_reply("ANSWERTEXT.")),
    )
    timed_ask(caplog, network, "QUESTIONTEXT about Anchorage?")
    logged = "\n".join(timing_events(caplog))
    assert "question_total" in logged
    for private in ("test-key", "QUESTIONTEXT", "Anchorage", "ANSWERTEXT", "GOOGLETEXT", "ANC", "3000", "threshold", "share_pct"):
        assert private not in logged


def test_timing_log_writes_to_stderr_with_one_handler(capsys):
    logger = gemini_agent.TIMING_LOGGER
    saved_handlers, saved_level = list(logger.handlers), logger.level
    logger.handlers.clear()
    try:
        gemini_agent.enable_timing_log()
        gemini_agent.enable_timing_log()
        assert len(logger.handlers) == 1
        gemini_agent.log_timing("example", duration_ms=5)
        assert "TIMING example duration_ms=5" in capsys.readouterr().err
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)


def test_timing_is_silent_until_enabled():
    logger = gemini_agent.TIMING_LOGGER
    saved_handlers, saved_level = list(logger.handlers), logger.level
    logger.handlers.clear()
    logger.setLevel("NOTSET")
    try:
        assert not logger.isEnabledFor(20)
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)
