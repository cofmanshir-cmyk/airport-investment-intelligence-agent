"""Decide from the data whether an answer rests on results retrieved in this conversation.

The airport catalog supplies every name, so no airport, region or question is special. Finding an
airport named in a question only checks what was retrieved. It never picks a tool or a metric.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from . import tools
from .agent import SYSTEM_PROMPT, ToolCall, find_unverified_numbers

# A real airport code that also appears in ordinary text about US airports.
NOT_AIRPORT_CODES = frozenset({"USA"})
GENERIC_NAME_ENDINGS = re.compile(r"(?:\s+(?:International|Regional|Municipal|Airport|Field))+$")


def _clean_name(name: str) -> str:
    return GENERIC_NAME_ENDINGS.sub("", name.strip())


@lru_cache(maxsize=None)
def _catalog() -> tuple[dict[str, str], dict[str, frozenset[str]], re.Pattern, re.Pattern]:
    """Map airport codes and names to T-100 codes. A shared city name maps to all of its airports."""
    codes: dict[str, str] = {}
    names: dict[str, set[str]] = {}
    for row in tools._airports().itertuples():
        for code in (row.airport, row.t100_code):
            if code not in NOT_AIRPORT_CODES:
                codes[code] = row.t100_code
        formal = str(row.airport_name)
        # A single word from a formal name is too often an ordinary word, such as a compass point.
        parts = [part for part in map(_clean_name, [formal, *re.split(r"[/-]", formal)]) if " " in part]
        for name in filter(None, [_clean_name(str(row.city)), *parts]):
            names.setdefault(name, set()).add(row.t100_code)

    def pattern(terms) -> re.Pattern:
        ordered = sorted(terms, key=len, reverse=True)
        return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ordered)) + r")(?!\w)")

    return codes, {name: frozenset(found) for name, found in names.items()}, pattern(codes), pattern(names)


def _coded_airports(text: str) -> set[str]:
    codes, _, code_pattern, _ = _catalog()
    return {codes[match] for match in code_pattern.findall(text)}


def _named_airports(text: str) -> dict[str, frozenset[str]]:
    _, names, _, name_pattern = _catalog()
    return {match: names[match] for match in name_pattern.findall(text)}


def _airports_in(result: object) -> set[str]:
    """Collect the airports a tool result reports on, from its structure rather than its notes."""
    found: set[str] = set()
    if isinstance(result, dict):
        if isinstance(result.get("airport"), str):
            found.add(result["airport"])
        if isinstance(result.get("airports"), dict):
            found.update(result["airports"])
        for value in result.values():
            found |= _airports_in(value)
    elif isinstance(result, list):
        for value in result:
            found |= _airports_in(value)
    return found


def retrieved_airports(calls: list[ToolCall]) -> set[str]:
    return {code for call in calls if not call.is_error for code in _airports_in(call.result)}


def _only_asks_for_clarification(text: str) -> bool:
    """A single question back to the user states nothing that needs data."""
    stripped = text.strip()
    return stripped.endswith("?") and not re.search(r"[.!?]\s", stripped) and not re.search(r"\d", stripped)


def _missing_labels(text: str, retrieved: set[str]) -> list[str]:
    labels = sorted(_coded_airports(text) - retrieved)
    labels += sorted(name for name, codes in _named_airports(text).items() if not codes & retrieved)
    return labels


def unsupported(question: str, text: str, calls: list[ToolCall], earlier: set[str], evidence: list[str]) -> list[str] | None:
    """Return what an answer lacks data for, or None if it is supported. An empty list means figures only.

    A code in the question must always be covered. Names are less exact, since "New York" is also a
    state, and answers may mention other airports as examples, so those checks apply only when this
    turn retrieved nothing. Relevance is checked by airport, not by measure.
    """
    if _only_asks_for_clarification(text):
        return None
    retrieved = earlier | retrieved_airports(calls)
    uncovered_codes = sorted(_coded_airports(question) - retrieved)
    if any(not call.is_error for call in calls):
        return uncovered_codes or None

    missing = list(dict.fromkeys([*_missing_labels(question, retrieved), *_missing_labels(text, retrieved)]))
    if missing:
        return missing
    # The model has the prompt and tool descriptions, so their figures are known to it.
    reference = [*evidence, SYSTEM_PROMPT, json.dumps(tools.TOOL_DEFINITIONS)]
    return [] if find_unverified_numbers(text, reference) else None


def withheld_message(missing: list[str]) -> str:
    subject = f" for {', '.join(missing)}" if missing else " to support this answer"
    return (
        f"I couldn't retrieve data{subject}, so I'm not showing an answer that the data doesn't support. "
        "You can ask again."
    )
