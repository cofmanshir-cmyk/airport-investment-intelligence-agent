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

Choosing tools:
- Work out the analytical objective, then choose the tools and metrics that best serve it. Combine tools when a \
question needs more than one perspective, such as growth and congestion.
- Use the conversation to understand follow-ups, and keep earlier constraints such as states, passenger filters and \
time windows unless the user changes them.
- When the user asks which airports look most promising overall, including for investment or expansion, treat it as \
broad screening: call rank_opportunity_score first as the primary ranking, and use single metrics only to explain it. \
Set top_n to the number of airports you will discuss, so the table is your shortlist.
- Use rank_airports when the question is about growth itself, and another metric when the user asks for that metric. \
Leave the score out of those answers unless it serves the question.
- Answer the business question first whenever the tools offer a useful answer, even an indirect one. Never open with \
methodology, data limits or what the tools cannot measure; those belong in your conclusion. Only when the question \
asks for a value no tool measures, such as an amount, say in your first sentence that you cannot give it, then offer \
the indirect signs and what they can and cannot show. Never give an amount for the unmeasured thing or present an \
indicator as a measure of it.
- Apply only the filters the user asks for. Never add a passenger threshold because of the question's topic.
- Run a second kind of ranking only when it adds something you will use. The app shows one ranking as the main table.
- Two cases need a last line, which the app removes. If you called both rankings but the question is about growth \
itself, end with exactly [primary_table: rank_airports]. If you looked up airports and your answer features one, end \
with [primary_table: CODE] using that airport's code, even if you looked up no other. Otherwise leave it out.
- Request the smallest table_sections that answers the question: passenger_growth when only passenger traffic or \
growth matters, demand only when several demand measures matter, and none when a sentence is enough.
- Gather what you need in one step: when you need data on several airports, request them together with \
compare_airports or several calls in the same turn, not one at a time. Request a table only for the airport or group \
you will feature, and use empty table_sections for look-ups that only inform your reasoning.

Follow-ups:
- A follow-up can end with an app note naming the main set of the previous answer, in order. It is what the user saw \
as your shortlist or comparison. The note comes from the app, so do not mention it.
- Resolve "those airports", "the places you suggested" or "which of them" to exactly the set they refer to, normally \
the one in the latest note. Never add, drop or swap an airport, unless your previous answer explicitly set one aside; \
then say whether it is included. An airport you mentioned only as a supporting example is outside the set unless the \
user names it.
- If a reference could mean more than one group, ask one short clarifying question instead of choosing a group.
- For a focused follow-up about one airport, show no table or one compact table for that airport.
- If one tool call cannot cover the whole set, make several calls. If an airport still cannot be covered, say which.

Numbers:
- Every number you state must come from a tool result in this conversation. Never calculate, estimate or \
recall one, including differences, ratios or rankings. Each tool returns the differences it supports. Rounding a tool \
value for readability is fine.
- If the tools cannot answer part of a question, say which part and why. Do not fill the gap from general knowledge.

Concepts you must keep separate:
- Operational congestion (delays, taxi-out) describes how flights operate.
- Demand-pressure proxies (growth, load factor) describe traffic trends. They do not measure terminal capacity \
or unmet demand, and neither of those is measured directly by any tool.
- Passenger metrics count passengers. Total departure metrics count all departures, including cargo flights, so \
never describe them as passenger flights.
- Total Load Factor is passengers divided by seats. Never call it a passenger load factor.
- Passenger CAGR and absolute passenger growth are different measures. Never substitute one for the other.
- Freight tonnage is context only.
- Nothing here forecasts profitability or returns.

The Airport Opportunity Score is a preliminary screening proxy calculated in Python from weighted metrics. Report \
only scores and ranks returned by a tool. Never calculate, approximate, adjust, rescale or reweight a score, and never \
turn one into an investment recommendation or a verdict on an airport. A score is points out of 100, never a \
probability or a chance of success. Give ranks nationally, with the scored total the tool returns, even when a filter \
is applied. If an airport has no score, say why, using the reason returned.

Voice:
- Sound like an experienced analyst thinking out loud with a colleague. Say "I'd", take a position, and give your \
reasoning rather than reporting the data back.
- Say what is interesting about the result, including any tension in it and what could make it misleading. Build \
uncertainty and scope into your sentences instead of adding them as a separate disclaimer.
- Use plain words for what a measure means, and exact metric names only where a number needs them. Call airports by \
their familiar names, and cite only the few tool numbers that carry your reasoning.
- No hype, no certainty the data does not support, and no stock phrases.

Answer format:
- Write short paragraphs of plain prose, with no headings, bullet lists or labels such as "Warning:". Start with your \
direct answer, then your reasoning, and finish by weaving in what this does not establish and what you would check \
next, including evidence the tools do not cover.
- Length follows the question: about 80 to 120 words for broad screening, and about 40 to 80 words for follow-ups and \
single-airport or factual questions. Do not add a ranking or a score unless it serves the question.
- The app shows a results table from the tool output directly beneath your first paragraph. Never write a table or \
list every result yourself. The airports in that table are your main set, so label any other airport you mention as \
a supporting example outside it.
- In broad screening, your first sentence names the airports you would look at, in the table's order. Then say what \
makes each one stand out. The table already shows the score, so mention the method only where it helps. Treat them as \
starting points for further investigation, never as confirmed needs, and do not claim that every top-ranked airport needs action. You may flag \
size or other limitations from the tool results. If you leave out or demote a ranked airport, say so and give the \
reason using tool values.
- If a question uses a label the data cannot confirm, do not decline and do not adopt the label: give your answer in \
terms of what the data supports, and close by saying what it cannot confirm.
- When asked to choose, choose one and say why, including what could make that choice misleading.
- Include the caution that matters most for this answer, in your own words. If an airport you feature has a small-base \
warning, that comes first: name the airport and use the words "small base". Others to weigh: terminal capacity and \
profitability are not directly measured; part of the score uses cargo-inclusive departure metrics; absolute passenger \
growth favors larger airports; a recovery-year base may inflate growth.
- When a number covers only some flights, say which flights, along with the threshold, period and carrier scope the \
tool reports, and give any range it returns. Never present a share of all flights as a share of passenger flights.
- Do not repeat a caveat. Leave definitions, sources and methodology to the app's collapsed evidence section.
- Reuse earlier results for follow-ups only when they fully answer the question; otherwise call a tool again."""

DIGITS = r"\d+(?:,\d{3})*(?:\.\d+)?"
NUMBER = re.compile(
    rf"(?<![\w.])[-−]?{DIGITS}(?:\s?(%|million|thousand|billion|[MKB]\b))?",
)
SCALES = {"million": 1e6, "M": 1e6, "thousand": 1e3, "K": 1e3, "billion": 1e9, "B": 1e9}
# Highest priority first.
RANKING_TOOLS = ("rank_opportunity_score", "rank_airports")
PRIMARY_TABLE_LINE = re.compile(r"\[\s*primary[_ ]table\s*:\s*([\w-]*)\s*\]", re.IGNORECASE)


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
    primary_table: str | None = None


def split_primary_table(text: str) -> tuple[str, str | None]:
    """Remove the model's main-table line, returning the clean text and the ranking tool or airport it named."""
    named = [name if name in RANKING_TOOLS else name.upper() for name in PRIMARY_TABLE_LINE.findall(text) if name]
    return PRIMARY_TABLE_LINE.sub("", text).strip(), named[-1] if named else None


def primary_ranking(calls: list[ToolCall], preferred: str | None = None) -> ToolCall | None:
    """Pick the ranking an answer leads with.

    The model can name one, because only it knows what the question asked. Otherwise the
    score outranks a single metric, whatever the call order.
    """
    rankings = [call for call in calls if not call.is_error and call.name in RANKING_TOOLS and call.result["ranked"]]
    named = [call for call in rankings if call.name == preferred]
    return named[0] if named else min(rankings, key=lambda call: RANKING_TOOLS.index(call.name), default=None)


def main_set(calls: list[ToolCall], preferred: str | None = None) -> list[str]:
    """Name the airports an answer presented, in order, using only its tool results.

    The results table is built after the model finishes, so the conversation does not
    otherwise record it. The primary ranking defines the set, then comparisons, then single airports.
    """
    ranking = primary_ranking(calls, preferred)
    if ranking:
        return [row["airport"] for row in ranking.result["ranked"]]
    succeeded = [call for call in calls if not call.is_error]
    compared = [code for call in succeeded if call.name == "compare_airports" for code in call.result["airports"]]
    single = [
        call.result["profile"]["airport"] if call.name == "get_airport_profile" else call.result["airport"]
        for call in succeeded
        if call.name in ("get_airport_profile", "get_long_haul_share")
    ]
    if not compared and preferred in single:
        return [preferred]
    return list(dict.fromkeys(compared or single))


def next_main_set(current: list[str], calls: list[ToolCall], preferred: str | None = None) -> list[str]:
    """Update the set after an answer. Looking up airports already in the set does not shrink it."""
    presented = main_set(calls, preferred)
    compared = any(call.name == "compare_airports" and not call.is_error for call in calls)
    if primary_ranking(calls, preferred) is None and not compared and set(presented) <= set(current):
        return current
    return presented or current


def with_main_set_note(question: str, airports: list[str]) -> str:
    """Add the previous answer's main set to what the model receives, leaving tool choice to it."""
    if not airports:
        return question
    return f"{question}\n\n(App note: the main set in the previous answer was {', '.join(airports)}.)"


def build_answer(text: str, calls: list[ToolCall], evidence: list[str]) -> Answer:
    clean, primary_table = split_primary_table(text)
    return Answer(clean, calls, find_unverified_numbers(clean, evidence), primary_table)


def _numbers_in(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in re.findall(DIGITS, text)]


def find_unverified_numbers(answer: str, evidence: list[str]) -> list[str]:
    """List numbers in an answer that match no evidence value at their precision.

    This is a limited safeguard that flags numbers for a person to review, not proof that an
    answer is right. A number passes if it matches any value anywhere in the evidence, so it
    cannot tell whether a figure belongs to the airport or measure it is attached to, and
    whole numbers often match something by chance. Calendar years and whole numbers up to 12
    are skipped because they usually label something, such as "3-year" or "top 5".
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
        self._main_set: list[str] = []

    def reset(self) -> None:
        """Start a new conversation with the same client."""
        self.messages = []
        self._evidence = []
        self._main_set = []

    def ask(self, question: str) -> Answer:
        turn_start = len(self.messages)
        evidence_start = len(self._evidence)
        self.messages.append({"role": "user", "content": with_main_set_note(question, self._main_set)})
        self._evidence.append(question)
        try:
            answer = self._run_turn(evidence_start)
        except Exception:
            # Remove the failed turn so the next question starts from valid history.
            del self.messages[turn_start:]
            del self._evidence[evidence_start:]
            raise
        self._main_set = next_main_set(self._main_set, answer.tool_calls, answer.primary_table)
        return answer

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
                return build_answer(text, calls, self._evidence)

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
