"""LLM conversation loop. The LLM chooses tools and explains; Python calculates."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import anthropic

from . import tools

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_TOKENS = 16_000
MAX_TOOL_ROUNDS = 6
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are an airport investment screening analyst working for a human analyst. \
Answer questions about US airports using the tools provided.

Numbers:
- Every number you state must come from a tool result in this conversation. Never calculate, estimate or \
recall one, including differences, ratios or rankings. Each tool returns the differences it supports.
- If the tools cannot answer part of a question, say which part and why. Do not fill the gap from general knowledge.

Concepts you must keep separate:
- Operational congestion (delays, taxi-out) describes how flights operate.
- Demand-pressure proxies (growth, load factor) describe traffic trends. They do not measure terminal capacity \
or unmet demand, and neither of those is measured directly by any tool.
- Passenger metrics count passengers. Total departure metrics count all departures, including cargo flights, so \
never describe them as passenger flights.
- Total Load Factor is passengers divided by seats. Never call it a passenger load factor.
- Passenger CAGR and absolute passenger growth are different measures. Show both when a score is discussed, and \
never substitute one for the other.
- Freight tonnage is context only.
- Nothing here forecasts profitability or returns.

The Airport Opportunity Score is a preliminary screening proxy calculated in Python from 11 weighted metrics. Report \
only scores and ranks returned by a tool. Never calculate, approximate, adjust or reweight a score, and never turn \
one into an investment recommendation or label an airport a strong candidate.

Whenever you report a score or score-based rank, also give the airport's 2022 and 2025 passengers, absolute passenger \
growth and Passenger CAGR 3Y as returned, and state that: 35% of the score uses cargo-inclusive departure metrics; \
absolute passenger growth favors larger airports; 2022 was a recovery year, which may inflate 3-year growth. State any \
small-base warning for an airport you mention. If an airport has no score, say why, using the reason returned.

Every answer must state, briefly: the result, how the measure is defined, the years covered, the data sources, and \
the caveats that matter for this question. Say plainly when a figure is provisional. Report airport warnings from \
tool results when they affect the answer. Reuse earlier results for follow-ups only when they fully answer the \
question; otherwise call a tool again."""

DIGITS = r"\d+(?:,\d{3})*(?:\.\d+)?"
NUMBER = re.compile(
    rf"(?<![\w.])[-−]?{DIGITS}(?:\s?(%|million|thousand|billion|[MKB]\b))?",
)
SCALES = {"million": 1e6, "M": 1e6, "thousand": 1e3, "K": 1e3, "billion": 1e9, "B": 1e9}


@dataclass
class ToolCall:
    name: str
    tool_input: dict
    result: dict
    is_error: bool


@dataclass
class Answer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)


def _numbers_in(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in re.findall(DIGITS, text)]


def find_unverified_numbers(answer: str, evidence: list[str]) -> list[str]:
    """List numbers in an answer that match no evidence value at their precision.

    This check flags numbers for a person to review. Calendar years and small
    whole numbers are skipped because they usually label something, such as
    "3-year" or "top 5".
    """
    known = [number for text in evidence for number in _numbers_in(text)]
    unverified = []
    for match in NUMBER.finditer(answer):
        token = match.group(0).lstrip("-−")
        digits = re.match(DIGITS, token).group(0)
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = SCALES.get(match.group(1) or "", 1)
        if decimals == 0 and scale == 1 and (value <= 12 or (1990 <= value <= 2030 and "," not in digits)):
            continue
        if not any(round(number / scale, decimals) == value for number in known):
            unverified.append(match.group(0))
    return list(dict.fromkeys(unverified))


class Agent:
    """Hold one conversation so follow-up questions keep their context."""

    def __init__(self, client: anthropic.Anthropic, model: str = MODEL):
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self._evidence: list[str] = []

    def ask(self, question: str) -> Answer:
        turn_start = len(self.messages)
        evidence_start = len(self._evidence)
        self.messages.append({"role": "user", "content": question})
        self._evidence.append(question)
        try:
            return self._run_turn(evidence_start)
        except Exception:
            # Remove the failed turn so the next question starts from valid history.
            del self.messages[turn_start:]
            del self._evidence[evidence_start:]
            raise

    def _run_turn(self, evidence_start: int) -> Answer:
        calls: list[ToolCall] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": "medium"},
                system=SYSTEM_PROMPT,
                tools=tools.TOOL_DEFINITIONS,
                messages=self.messages,
            )

            if response.stop_reason == "refusal":
                raise AgentError("The model declined this request. Try rephrasing the question.")
            if response.stop_reason == "max_tokens":
                raise AgentError("The answer exceeded the length limit. Try a narrower question.")

            # The full content goes back unchanged, as the API requires.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                text = "\n".join(block.text for block in response.content if block.type == "text").strip()
                return Answer(
                    text=text,
                    tool_calls=calls,
                    unverified_numbers=find_unverified_numbers(text, self._evidence),
                )

            results = []
            for block in tool_uses:
                result, is_error = tools.execute_tool(block.name, block.input)
                content = tools.to_json(result)
                calls.append(ToolCall(block.name, block.input, result, is_error))
                self._evidence.append(content)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
            self.messages.append({"role": "user", "content": results})
        raise AgentError(f"No answer after {MAX_TOOL_ROUNDS} tool rounds. Try a narrower question.")


class AgentError(RuntimeError):
    """A problem to show the user directly, without a stack trace."""
