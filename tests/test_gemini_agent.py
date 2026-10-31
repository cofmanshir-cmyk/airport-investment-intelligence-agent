"""Gemini agent checks using the real SDK over a fake network.

No request leaves this machine and no API allowance is used.
"""

import base64
import json

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
    assert "Airport Opportunity Score is not implemented" in body["systemInstruction"]["parts"][0]["text"]
    assert body["generationConfig"]["maxOutputTokens"] > 0


def test_integer_enums_move_to_the_description_and_tools_stay_unchanged():
    before = json.dumps(tools.TOOL_DEFINITIONS, sort_keys=True)
    schemas = {tool.name: tool.parameters_json_schema for tool in gemini_agent.tool_declarations()[0].function_declarations}
    window = schemas["rank_airports"]["properties"]["window_years"]
    assert "enum" not in window
    assert "Allowed values: 3, 10" in window["description"]
    assert "additionalProperties" not in schemas["rank_airports"]
    assert json.dumps(tools.TOOL_DEFINITIONS, sort_keys=True) == before


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


def test_session_limit_stops_before_making_a_call():
    network = FakeNetwork(text_reply("One."), text_reply("Two."))
    agent = make_agent(network, call_limit=1)
    agent.ask("First")
    with pytest.raises(AgentError, match="session limit"):
        agent.ask("Second")
    assert len(network.bodies) == 1
    assert agent.calls_used == 1


def test_endless_tool_calls_stop_at_the_round_limit():
    replies = [function_call("get_airport_profile", {"airport": "SFO"}) for _ in range(gemini_agent.MAX_TOOL_ROUNDS)]
    network = FakeNetwork(*replies)
    with pytest.raises(AgentError, match="tool rounds"):
        make_agent(network).ask("Question")
    assert len(network.bodies) == gemini_agent.MAX_TOOL_ROUNDS


def test_quota_error_makes_one_call_and_leaves_history_clean():
    network = FakeNetwork({"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}, status=429)
    agent = make_agent(network)
    with pytest.raises(errors.ClientError):
        agent.ask("Question")
    assert len(network.bodies) == 1
    assert agent.contents == []
    assert agent.calls_used == 1


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
    assert agent.calls_used == 1


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


def test_model_selection_prefers_newest_available():
    assert select_model(make_client(FakeNetwork())) == "gemini-3.8-flash"


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
