"""Agent loop checks using a scripted client, so no API key or network is needed.

The scripted client stands in for the LLM in tests only; the application always
uses the real API.
"""

from types import SimpleNamespace

import pytest

from src import config, tools
from src.agent import Agent, AgentError, MAX_TOOL_ROUNDS, find_unverified_numbers

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
