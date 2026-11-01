"""Shape tool results for display. Values are copied, never calculated here."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field, replace

import pandas as pd

from . import scoring, tools
from .agent import NUMBER, RANKING_TOOLS, ToolCall, find_unverified_numbers, primary_ranking

PROVISIONAL_PREFIX = "PROVISIONAL"
SMALL_BASE_PREFIX = "Small base"
UNVERIFIED_FIGURE = "[figure not verified]"

TOOL_LABELS = {
    "rank_opportunity_score": "Airport Opportunity Score",
    "rank_airports": "Passenger growth ranking",
    "compare_airports": "Airport comparison",
    "get_airport_profile": "Airport profile",
    "get_long_haul_share": "Long-haul share",
}

LABELS = {
    "passenger_cagr_3y_pct": "Passenger CAGR 3Y (%)",
    "passenger_cagr_10y_pct": "Passenger CAGR 10Y (%)",
    "absolute_passenger_growth_3y": "Absolute Passenger Growth 3Y (passengers)",
    "absolute_passenger_growth_10y": "Absolute Passenger Growth 10Y (passengers), not scored",
    "flight_cagr_3y_pct": "Total Departure CAGR 3Y (%), incl. cargo",
    "flight_cagr_10y_pct": "Total Departure CAGR 10Y (%), incl. cargo",
    "growth_gap_3y_pts": "Passenger vs Total Departure Growth Gap 3Y (points), incl. cargo",
    "total_load_factor_2025_pct": "Total Load Factor 2025 (%), passengers / seats",
    "load_factor_change_3y_pts": "Total Load Factor Change 3Y (points), passengers / seats",
    "passengers_per_flight_growth_3y_pct": "Passengers per Departure Growth 3Y (%), incl. cargo",
    "flight_growth_acceleration_pts": "Total Departure Growth Acceleration (points), incl. cargo",
    "passenger_growth_acceleration_pts": "Passenger Growth Acceleration (points)",
    "opportunity_score": "Airport Opportunity Score",
    "national_rank": "National rank",
    "departures_in_sample": "Departures in sample",
    "share_of_t100_domestic_departures_pct": "Share of T-100 domestic departures (%)",
    "departures_delayed_15_plus_min_pct": "Departures delayed 15+ min (%)",
    "cancelled_departures_pct": "Cancelled departures (%)",
    "average_taxi_out_min": "Average taxi-out (min)",
    "arrivals_delayed_15_plus_min_pct": "Arrivals delayed 15+ min (%)",
    "nas_delay_min_per_arrival": "NAS delay (min per scheduled arrival)",
    "weather_delay_min_per_arrival": "Weather delay (min per scheduled arrival)",
    "carrier_delay_min_per_arrival": "Carrier delay (min per scheduled arrival)",
    "late_aircraft_delay_min_per_arrival": "Late-aircraft delay (min per scheduled arrival)",
}

RANK_COLUMNS = {
    "national_rank": "National rank",
    "airport": "Airport",
    "airport_name": "Name",
    "state": "State",
    "faa_hub_class": "FAA hub class",
    "faa_enplanements_2025": "FAA enplanements 2025",
}

SCORE_LABEL = "Score (points out of 100)"
SCORE_TABLE_TITLE = "Airport Opportunity Score"
DEMAND_TABLE_TITLE = "Demand-pressure proxies"
SCORE_PROXY_NOTE = "Preliminary screening proxy, not an investment recommendation."
POSITION_LABEL = "Position in this list"
_MAIN_TABLE_HEADINGS = {
    "Operational congestion": "Congestion indicators",
    DEMAND_TABLE_TITLE: "Demand indicators (proxies)",
    "Cargo context, not scored": "Cargo context (not scored)",
}

SCORE_COLUMNS = {
    "national_rank": "National rank",
    "position_in_this_list": POSITION_LABEL,
    "airport": "Airport",
    "airport_name": "Name",
    "state": "State",
    "faa_hub_class": "FAA hub class",
    "opportunity_score": SCORE_LABEL,
    "passengers_2022": "Passengers 2022",
    "passengers_2025": "Passengers 2025",
    "absolute_passenger_growth_3y": "Absolute Passenger Growth 3Y",
    "passenger_cagr_3y_pct": "Passenger CAGR 3Y (%)",
    "base_year_warning": "Base-year warning",
}

SCOPE_LABELS = {
    "all": "All flights in file",
    "passenger": "Passenger aircraft",
    "scheduled_passenger": "Scheduled passenger aircraft",
    "freighter": "Freighters",
}


@dataclass
class Table:
    title: str
    data: pd.DataFrame
    note: str = ""


@dataclass
class ToolEvidence:
    tool: str
    label: str
    inputs: dict
    error: str | None = None
    periods: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    definitions: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    provisional: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def _section_table(title: str, columns: dict[str, dict | None], note: str = "") -> Table | None:
    """Build a measure-by-airport table, leaving out empty sections."""
    columns = {name: values for name, values in columns.items() if values}
    if not columns:
        return None
    frame = pd.DataFrame(columns).rename(index=LABELS)
    frame.index.name = "Measure"
    return Table(title, frame.reset_index(), note)


def _rank_label(rank: int | None, total: int) -> str:
    return "n/a" if rank is None else f"#{rank} of {total}"


def _display(value, pattern: str = "{:,}") -> str:
    return "n/a" if value is None else pattern.format(value)


def _score_summary(section: dict) -> dict:
    """Format each value as text, because this table mixes scores, ranks and counts."""
    return {
        SCORE_LABEL: _display(section["opportunity_score"], "{:.2f}"),
        "National rank": _rank_label(section["national_rank"], section["national_peer_count"]),
        "Passengers 2022": _display(section["passengers_2022"]),
        "Passengers 2025": _display(section["passengers_2025"]),
        "Absolute Passenger Growth 3Y": _display(section["absolute_passenger_growth_3y"]),
        "Passenger CAGR 3Y (%)": _display(section["passenger_cagr_3y_pct"], "{:.2f}"),
    }


def _score_tables(profiles: dict[str, dict]) -> list[Table]:
    """Show the score with its passenger context, then the points behind it."""
    sections = {code: profile["opportunity_score"] for code, profile in profiles.items()}
    reasons = [f"{code} has no score: {section['reason']}." for code, section in sections.items() if "reason" in section]
    tables = [_section_table(
        SCORE_TABLE_TITLE,
        {code: _score_summary(section) for code, section in sections.items()},
        " ".join([SCORE_PROXY_NOTE, *reasons]),
    )]
    points = {
        code: {metric: part["points"] for metric, part in section["components"].items()}
        for code, section in sections.items()
        if "components" in section
    }
    if points:
        tables.append(_section_table("Score points by metric", points, "Points = weight x normalized value / 100."))
    return tables


def _airport_sections(profiles: dict[str, dict]) -> dict[str, list[Table]]:
    """Keep the score, demand, congestion and cargo in separate tables."""
    congestion = {code: profile["operational_congestion"] for code, profile in profiles.items()}
    congestion_rows = {
        code: {key: value for key, value in section.items() if key not in ("year", "months")} if section else None
        for code, section in congestion.items()
    }
    years = {section["year"] for section in congestion.values() if section}
    year_note = f"Calendar {years.pop()}." if len(years) == 1 else ""
    cargo = {code: profile["cargo_context_not_scored"] for code, profile in profiles.items()}
    sections = {
        "score": _score_tables(profiles),
        "demand": [_section_table(
            DEMAND_TABLE_TITLE,
            {
                code: {
                    **profile["demand_pressure_proxies"]["metrics"],
                    **profile["demand_pressure_proxies"]["supplementary_not_scored"],
                }
                for code, profile in profiles.items()
            },
            "Raw values, not normalized. Total departure metrics include cargo flights. They do not measure "
            "congestion, terminal capacity, unmet demand or profitability. Blank means a required year is incomplete.",
        )],
        "congestion": [_section_table(
            "Operational congestion",
            congestion_rows,
            f"{year_note} Reporting carriers on domestic flights only.".strip(),
        )],
        "cargo": [_section_table(
            "Cargo context, not scored",
            {
                code: {"Freight (lbs)": section["freight_lbs"]} if section else None
                for code, section in cargo.items()
            },
        )],
    }
    return {name: [table for table in tables if table is not None] for name, tables in sections.items()}


def _airport_tables(profiles: dict[str, dict]) -> list[Table]:
    return [table for tables in _airport_sections(profiles).values() for table in tables]


def _filter_note(filters: dict) -> str:
    states = filters["states"]
    states_text = "all states" if states == "all" else ", ".join(states)
    return f"Shown: {filters['passengers_2025']}; {states_text}. Filters do not change scores or national ranks."


def _annual_inputs_table(airport: str, profile: dict) -> Table:
    inputs = profile["demand_pressure_proxies"]["annual_inputs"]
    frame = pd.DataFrame(inputs).T.rename(columns=lambda name: name.replace("_", " ").capitalize())
    frame.index.name = "Year"
    return Table(f"{airport} annual inputs", frame.reset_index(), "Annual API totals behind the metrics.")


def _rank_evidence(result: dict) -> tuple[list[Table], list[str]]:
    base, end, window = result["base_year"], result["end_year"], result["window_years"]
    absolute = result["rank_by"] == "absolute_passenger_growth"
    growth = {
        "absolute_passenger_growth": f"Absolute Passenger Growth {window}Y",
        "passenger_cagr_pct": f"Passenger CAGR {window}Y (%)",
    }
    # The ranked measure comes first after the airport's identity.
    measures = list(growth.items()) if absolute else list(reversed(growth.items()))
    columns = {
        "national_rank": "National rank",
        **({"position_in_this_list": POSITION_LABEL} if any("position_in_this_list" in row for row in result["ranked"]) else {}),
        "airport": "Airport",
        "airport_name": "Name",
        "state": "State",
        "faa_hub_class": "FAA hub class",
        **dict(measures),
        f"passengers_{base}": f"Passengers {base}",
        f"passengers_{end}": f"Passengers {end}",
    }
    tables = []
    if result["ranked"]:
        total = result["airports_in_national_ranking"]
        rows = [{**row, "national_rank": _rank_label(row["national_rank"], total)} for row in result["ranked"]]
        ranked = pd.DataFrame(rows).reindex(columns=list(columns)).rename(columns=columns)
        tables.append(Table(
            f"Ranked by Absolute Passenger Growth {window}Y" if absolute else f"Ranked by Passenger CAGR {window}Y",
            ranked,
            f"One raw metric, not an investment ranking. {_filter_note(result['filters'])}"
            + (" Absolute growth tends to favor larger airports." if absolute else ""),
        ))
    if result["not_ranked"]:
        not_ranked = pd.DataFrame(result["not_ranked"]).reindex(columns=["airport", "airport_name", "state", "reason"])
        tables.append(Table(
            "Not ranked",
            not_ranked.rename(columns={"airport": "Airport", "airport_name": "Name", "state": "State", "reason": "Reason"}),
            "Shown instead of estimating missing data.",
        ))
    warnings = [f"{row['airport']}: {warning}" for row in result["ranked"] for warning in row["warnings"]]
    return tables, warnings


def _opportunity_evidence(result: dict) -> tuple[list[Table], list[str]]:
    tables, warnings = [], []
    if result["ranked"]:
        total = result["airports_scored_nationally"]
        rows = [
            {
                **row,
                "national_rank": _rank_label(row["national_rank"], total),
                "base_year_warning": "Small base" if row["small_base_warnings"] else "",
            }
            for row in result["ranked"]
        ]
        columns = {
            key: label
            for key, label in SCORE_COLUMNS.items()
            if key != "position_in_this_list" or "position_in_this_list" in rows[0]
        }
        tables.append(Table(
            SCORE_TABLE_TITLE,
            pd.DataFrame(rows).reindex(columns=list(columns)).rename(columns=columns),
            f"{SCORE_PROXY_NOTE} {_filter_note(result['filters'])}",
        ))
        points = {row["airport"]: row["score_points"] for row in result["ranked"]}
        tables.append(_section_table("Score points by metric", points, "Points = weight x normalized value / 100."))
        for row in result["ranked"]:
            warnings += [f"{row['airport']}: {warning}" for warning in [*row["small_base_warnings"], *row["warnings"]]]
    if result["not_scored"]:
        not_scored = pd.DataFrame(result["not_scored"]).reindex(columns=["airport", "airport_name", "state", "reason"])
        tables.append(Table(
            "No score",
            not_scored.rename(columns={"airport": "Airport", "airport_name": "Name", "state": "State", "reason": "Reason"}),
            "These airports have no score because a required metric is unavailable. Nothing is estimated or reweighted.",
        ))
    return tables, warnings


def methodology_table() -> pd.DataFrame:
    """List the score's metrics and weights directly from the scoring module."""
    return pd.DataFrame({
        "Measure": [LABELS[metric] for metric in scoring.WEIGHTS],
        "Weight (%)": list(scoring.WEIGHTS.values()),
    })


def small_base_alerts(calls: list[ToolCall]) -> list[str]:
    """Collect full small-base warnings for the evidence section."""
    alerts = [
        warning
        for call in calls
        for warning in build_evidence(call).warnings
        if SMALL_BASE_PREFIX in warning
    ]
    return list(dict.fromkeys(alerts))


def split_answer(text: str) -> tuple[str, str]:
    """Separate the analysis from the closing note, so the tables can sit between them.

    The last paragraph is the note, and everything before it stays together above the tables. A
    single paragraph is all analysis. Results tables come only from tool output, so any table the
    model writes is removed.
    """
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("|")]
    parts = [part.strip() for part in re.split(r"\n\s*\n", "\n".join(lines).strip()) if part.strip()]
    if len(parts) < 2:
        return (parts[0] if parts else ""), ""
    return "\n\n".join(parts[:-1]), parts[-1]


def hide_unverified_numbers(text: str, unverified: list[str]) -> str:
    """Replace flagged numbers for display, so the text never states them as facts.

    No value is substituted, because the check can't tell which returned value was meant. The stored
    answer keeps the original text, and surrounding words stay as written.
    """
    return NUMBER.sub(lambda match: UNVERIFIED_FIGURE if match.group(0) in unverified else match.group(0), text)


def _passenger_growth_table(profiles: dict[str, dict]) -> Table:
    """Show only the passenger growth measures. Evidence keeps every demand metric."""
    def measures(profile: dict) -> dict:
        demand = profile["demand_pressure_proxies"]
        metrics, inputs = demand["metrics"], demand["annual_inputs"]
        return {
            "Passengers 2022": _display(inputs["2022"].get("passengers")),
            "Passengers 2025": _display(inputs["2025"].get("passengers")),
            "Absolute Passenger Growth 3Y": _display(metrics["absolute_passenger_growth_3y"]),
            "Absolute Passenger Growth 10Y": _display(demand["supplementary_not_scored"]["absolute_passenger_growth_10y"]),
            "Passenger CAGR 3Y (%)": _display(metrics["passenger_cagr_3y_pct"], "{:.2f}"),
            "Passenger CAGR 10Y (%)": _display(metrics["passenger_cagr_10y_pct"], "{:.2f}"),
        }

    return _section_table(
        "Passenger growth",
        {code: measures(profile) for code, profile in profiles.items()},
        "Departing passengers in full calendar years; 10Y runs from 2015. n/a means a required year is incomplete. "
        "Absolute growth tends to favor larger airports.",
    )


def _named_at(code: str, profile: dict, text: str) -> int | None:
    """Find where a text first names an airport, by its code or the city in its tool result."""
    matches = [re.search(rf"\b{re.escape(code)}\b", text)]
    if profile.get("city"):
        matches.append(re.search(rf"\b{re.escape(profile['city'])}\b", text, re.IGNORECASE))
    positions = [match.start() for match in matches if match]
    return min(positions) if positions else None


def _featured_airport(calls: list[ToolCall], narrative: str, preferred: str | None) -> str | None:
    """Pick the looked-up airport the answer features.

    The model's own line decides, because the first airport named may only be a contrast. An airport
    it names but did not look up gets no table. Without the line, the first one named is the best
    available guess, and several unnamed look-ups feature none.
    """
    profiles = {}
    for call in calls:
        if call.name == "get_airport_profile" and not call.is_error:
            profiles.setdefault(call.result["profile"]["airport"], call.result["profile"])
    if preferred and preferred not in RANKING_TOOLS:
        return preferred if preferred in profiles else None
    for text in split_answer(narrative):
        positions = {code: _named_at(code, profile, text) for code, profile in profiles.items()}
        named = {code: position for code, position in positions.items() if position is not None}
        if named:
            return min(named, key=named.get)
    return next(iter(profiles)) if len(profiles) == 1 else None


def _one_compact_table(tables: list[Table]) -> list[Table]:
    """Keep one table, passing over the long demand table when a shorter one was also requested."""
    return next(([table] for table in tables if table.title != DEMAND_TABLE_TITLE), tables[:1])


def _period_label(period: dict) -> str:
    """Turn a period such as 2026-01 to 2026-07 into January-July 2026."""
    first_year, first_month = period["first"].split("-")
    last_year, last_month = period["last"].split("-")
    first, last = calendar.month_name[int(first_month)], calendar.month_name[int(last_month)]
    if first_year != last_year:
        return f"{first} {first_year} to {last} {last_year}"
    return f"{first} {first_year}" if first == last else f"{first}\u2013{last} {first_year}"


def _long_haul_main_table(result: dict) -> Table:
    """Keep each share beside the flights it describes, with the limits that apply to it."""
    rows = [
        {
            "Flights": SCOPE_LABELS[scope],
            "Departures": int(result["scopes"][scope]["departures"]),
            "Long-haul departures": int(result["scopes"][scope]["long_haul_departures"]),
            "Long-haul share (%)": result["scopes"][scope]["share_pct"],
        }
        for scope in ("all", "passenger", "freighter")
    ]
    note = (
        f"Long-haul means at least {result['threshold_miles']:,g} statute miles, an assumption with no single standard "
        f"definition. {_period_label(result['period'])}, US carriers only."
    )
    coverage = result.get("coverage") or {}
    if coverage.get("available"):
        bounds = coverage["all_flights_share_bounds_pct"]
        note += (
            f" This file covers {coverage['departure_coverage_pct']:g}% of {result['airport']}'s departures, so the share of "
            f"all flights could be {bounds['low']:g}% to {bounds['high']:g}%."
        )
    else:
        note += " Coverage against all carriers could not be calculated."
    return Table(f"{result['airport']} long-haul share", pd.DataFrame(rows), note)


def main_tables(calls: list[ToolCall], preferred: str | None = None, narrative: str | None = None) -> list[Table]:
    """Choose the results tables shown with an answer. The evidence section always keeps everything.

    Different rankings would compete, so only the primary kind is shown. With the narrative, airport
    look-ups must also match what the answer discusses: a table for the featured airport only, and
    none when a ranking leads. A comparison that names just one of its airports keeps one table.
    """
    primary = primary_ranking(calls, preferred)
    featured = None if narrative is None else _featured_airport(calls, narrative, preferred)
    shown_airports = set()
    tables = []
    for call in calls:
        if call.is_error or (call.name in RANKING_TOOLS and (primary is None or call.name != primary.name)):
            continue
        result = call.result
        if call.name == "rank_opportunity_score":
            tables += _opportunity_evidence(result)[0][:1] if result["ranked"] else []
        elif call.name == "rank_airports":
            tables += _rank_evidence(result)[0][:1] if result["ranked"] else []
        elif call.name == "get_long_haul_share":
            tables.append(_long_haul_main_table(result))
        elif call.name in ("compare_airports", "get_airport_profile"):
            profiles = result["airports"] if call.name == "compare_airports" else {result["profile"]["airport"]: result["profile"]}
            sections = _airport_sections(profiles)
            # A comparison needs a table by default; one airport's fact often does not.
            chosen = result.get("table_sections")
            if chosen is None:
                chosen = list(sections) if call.name == "compare_airports" else []
            sections["passenger_growth"] = [_passenger_growth_table(profiles)]
            requested = [sections[name][0] for name in chosen if sections[name]]
            if narrative is None:
                tables += requested
            elif call.name == "get_airport_profile":
                code = result["profile"]["airport"]
                if primary is None and code == featured and code not in shown_airports:
                    shown_airports.add(code)
                    tables += _one_compact_table(requested)
            else:
                named = [code for code, profile in profiles.items() if _named_at(code, profile, narrative) is not None]
                focused = preferred in profiles or len(named) == 1
                tables += _one_compact_table(requested) if focused else requested
    return tables


def with_short_ranks(tables: list[Table]) -> list[Table]:
    """Show "#1" rather than "#1 of 229". The evidence tables keep the full form."""
    return [replace(table, data=table.data.replace(r"^(#\d+) of \d+$", r"\1", regex=True)) for table in tables]


def with_contextual_headings(tables: list[Table]) -> list[Table]:
    """Say what each airport table shows and for whom. Evidence and rankings keep their own titles."""
    def airports(codes: list[str]) -> str:
        return " and ".join(codes) if len(codes) < 3 else f"{', '.join(codes[:-1])} and {codes[-1]}"

    return [
        replace(table, title=f"{_MAIN_TABLE_HEADINGS.get(table.title, table.title)} for {airports(list(table.data.columns[1:]))}")
        if "Measure" in table.data.columns
        else table
        for table in tables
    ]


def _returned_scores(calls: list[ToolCall]) -> list[tuple[float, int, int]]:
    """List each returned score as (score, national rank, airports scored nationally)."""
    scores = []
    for call in calls:
        if call.is_error:
            continue
        result = call.result
        if call.name == "rank_opportunity_score":
            total = result["airports_scored_nationally"]
            scores += [(row["opportunity_score"], row["national_rank"], total) for row in result["ranked"]]
        elif call.name in ("compare_airports", "get_airport_profile"):
            profiles = result["airports"].values() if call.name == "compare_airports" else [result["profile"]]
            sections = [profile["opportunity_score"] for profile in profiles]
            scores += [
                (section["opportunity_score"], section["national_rank"], section["national_peer_count"])
                for section in sections
                if section["opportunity_score"] is not None
            ]
    return scores


def _states_returned_score(score: float, rank: int, total: int, narrative: str) -> bool:
    """Match an exact returned score, never a percentage, or a rank out of the scored total."""
    return bool(
        re.search(rf"(?<![\d.,]){re.escape(str(score))}0*(?!\d|\s?%)", narrative)
        or re.search(rf"(?<![\d.,])#?{rank} of {total}\b", narrative)
    )


def reports_opportunity_score(
    calls: list[ToolCall], narrative: str, earlier_calls: list[ToolCall] | None = None, preferred: str | None = None
) -> bool:
    """Decide from tool results and displayed values whether this answer reports the score.

    The profile and comparison tools always return score data, so returning it is not
    enough. Earlier results count because a follow-up can restate a score without a new tool.
    """
    succeeded = [call for call in calls if not call.is_error]
    primary = primary_ranking(succeeded, preferred)
    if primary is None and any(call.name == "rank_opportunity_score" for call in succeeded):
        return True
    if any(table.title == SCORE_TABLE_TITLE for table in main_tables(succeeded, preferred, narrative)):
        return True
    return any(
        _states_returned_score(score, rank, total, narrative)
        for score, rank, total in _returned_scores([*(earlier_calls or []), *succeeded])
    )


def without_score_proxy_note(tables: list[Table]) -> list[Table]:
    """Remove the caption that would repeat the answer's own score disclaimer."""
    return [replace(table, note=table.note.removeprefix(SCORE_PROXY_NOTE).strip()) for table in tables]


def small_base_notice(calls: list[ToolCall], narrative: str) -> str | None:
    """Return one short warning for small-base airports the narrative does not warn about."""
    paragraphs = re.split(r"\n\s*\n", narrative)
    years: dict[str, list[str]] = {}
    for alert in small_base_alerts(calls):
        match = re.match(r"(\w+): Small base year: (\d{4}) had ([\d,]+) passengers", alert)
        if match:
            years.setdefault(match[1], []).append(f"{match[2]}: {match[3]}")
    missing = [
        f"{code} ({'; '.join(values)} passengers)"
        for code, values in years.items()
        if not any(re.search(rf"\b{code}\b", part) and re.search(r"small[- ]base", part, re.I) for part in paragraphs)
    ]
    if not missing:
        return None
    return f"Small base year: {', '.join(missing)}. Percentage growth may be inflated."


def _long_haul_evidence(result: dict) -> tuple[list[Table], list[str]]:
    rows = []
    for scope, values in result["scopes"].items():
        rows.append({
            "Scope": SCOPE_LABELS.get(scope, scope),
            "Departures": values["departures"],
            "Long-haul departures": values["long_haul_departures"],
            f"Share at {result['threshold_miles']}+ miles (%)": values["share_pct"],
            **{f"Share at {miles}+ (%)": share for miles, share in values["sensitivity_pct"].items()},
        })
    tables = [Table(
        f"{result['airport']} long-haul share",
        pd.DataFrame(rows),
        f"Weighted by {result['weighting']}. Covers {result['period']['first']} to {result['period']['last']}.",
    )]
    warnings = []
    coverage = result.get("coverage") or {}
    if coverage.get("available"):
        bounds = coverage["all_flights_share_bounds_pct"]
        tables.append(Table("Route file coverage", pd.DataFrame([{
            "Departure coverage (%)": coverage["departure_coverage_pct"],
            "Passenger coverage (%)": coverage["passenger_coverage_pct"],
            "Departures missing from file": coverage["departures_missing_from_route_file"],
            "All-flight share, low bound (%)": bounds["low"],
            "All-flight share, high bound (%)": bounds["high"],
        }]), "Compared with the all-carrier API for the same months."))
    elif coverage:
        warnings.append(coverage.get("reason", "Coverage could not be calculated."))
    return tables, warnings


def build_evidence(call: ToolCall) -> ToolEvidence:
    """Convert one tool call into display sections."""
    evidence = ToolEvidence(
        tool=call.name,
        label=TOOL_LABELS.get(call.name, call.name),
        inputs=call.tool_input,
        raw=call.result,
    )
    if call.is_error:
        evidence.error = call.result.get("error", "The tool rejected this input.")
        return evidence

    result = call.result
    if call.name == "rank_opportunity_score":
        evidence.periods = [result["periods"]]
        evidence.tables, evidence.warnings = _opportunity_evidence(result)
    elif call.name == "rank_airports":
        evidence.periods = [result["periods"]]
        evidence.tables, evidence.warnings = _rank_evidence(result)
    elif call.name == "compare_airports":
        evidence.periods = list(result["periods"].values())
        profiles = result["airports"]
        evidence.tables = _airport_tables(profiles)
        evidence.warnings = [f"{code}: {warning}" for code, profile in profiles.items() for warning in profile["warnings"]]
    elif call.name == "get_airport_profile":
        evidence.periods = list(result["periods"].values())
        profile = result["profile"]
        evidence.tables = [*_airport_tables({profile["airport"]: profile}), _annual_inputs_table(profile["airport"], profile)]
        evidence.warnings = [f"{profile['airport']}: {warning}" for warning in profile["warnings"]]
    elif call.name == "get_long_haul_share":
        period = result["period"]
        evidence.periods = [f"{period['first']} to {period['last']} ({period['months']} months)"]
        evidence.tables, evidence.warnings = _long_haul_evidence(result)

    evidence.definitions = result.get("definitions", {})
    evidence.sources = result.get("sources", [])
    caveats = [*result.get("caveats", []), *result.get("assumptions", [])]
    evidence.provisional = [caveat for caveat in caveats if caveat.startswith(PROVISIONAL_PREFIX)]
    evidence.caveats = [caveat for caveat in caveats if not caveat.startswith(PROVISIONAL_PREFIX)]
    return evidence


_GROWTH_DEFINITIONS = (
    "passenger_cagr_3y_pct", "passenger_cagr_10y_pct", "absolute_passenger_growth_3y", "absolute_passenger_growth_10y",
)
_SECTION_TITLES = {
    "score": (SCORE_TABLE_TITLE, "Score points by metric"),
    "demand": (DEMAND_TABLE_TITLE,),
    "congestion": ("Operational congestion",),
    "cargo": ("Cargo context, not scored",),
}
_SECTION_DEFINITIONS = {
    "score": (*scoring.WEIGHTS, "opportunity_score", "national_rank"),
    "demand": (*scoring.WEIGHTS, "absolute_passenger_growth_10y"),
    "passenger_growth": _GROWTH_DEFINITIONS,
}
# Describes the ratios in analytics.congestion_metrics, which the tool returns without definitions.
CONGESTION_DEFINITIONS = {
    "departures_in_sample": "Scheduled domestic departures by reporting carriers in the on-time data, including cancelled flights",
    "share_of_t100_domestic_departures_pct": (
        "Departures in sample / T-100 domestic departures performed, which shows the data's coverage. "
        "The sample includes cancelled flights, so this can slightly overstate coverage and can exceed 100%"
    ),
    "departures_delayed_15_plus_min_pct": "Departures delayed 15+ minutes / departures with a delay status, percent",
    "cancelled_departures_pct": "Cancelled departures / departures in sample, percent",
    "average_taxi_out_min": "Total taxi-out minutes / departures with a taxi-out time",
    "arrivals_delayed_15_plus_min_pct": "Arrivals delayed 15+ minutes / arrivals with a delay status, percent",
    "nas_delay_min_per_arrival": "National Aviation System delay minutes / scheduled arrivals, including cancelled flights",
    "weather_delay_min_per_arrival": "Weather delay minutes / scheduled arrivals, including cancelled flights",
    "carrier_delay_min_per_arrival": "Carrier delay minutes / scheduled arrivals, including cancelled flights",
    "late_aircraft_delay_min_per_arrival": "Late-aircraft delay minutes / scheduled arrivals, including cancelled flights",
}
_GROWTH_CAVEATS = (tools.SIZE_BIAS, tools.BASE_2022, tools.PASSENGER_COUNT, tools.CONNECTING, tools.INCOMPLETE_YEARS)
_SECTION_CAVEATS = {
    "score": (
        *_GROWTH_CAVEATS, tools.CARGO_INCLUSIVE, tools.LOAD_FACTOR, tools.METRIC_OVERLAP, tools.THRESHOLD,
    ),
    "demand": (*_GROWTH_CAVEATS, tools.LOAD_FACTOR),
    "passenger_growth": _GROWTH_CAVEATS,
    "congestion": (tools.CONGESTION_SCOPE,),
}
_SECTION_SOURCES = {
    "score": ("t100_summary", "faa", "t100_routes"),
    "demand": ("t100_summary",),
    "passenger_growth": ("t100_summary",),
    "congestion": ("ontime", "t100_summary"),
    "cargo": ("t100_summary",),
}
# Cheapest explanation first, so a growth figure never brings in the score table.
_SECTIONS_BY_SPECIFICITY = ("congestion", "cargo", "passenger_growth", "demand", "score")


def _table_text(tables: list[Table]) -> str:
    return "\n".join(table.data.to_csv(index=False) for table in tables)


def _answer_sections(call: ToolCall, narrative: str) -> list[str]:
    """List the sections behind an answer: those requested, plus any whose numbers the answer cites."""
    result = call.result
    profiles = result["airports"] if call.name == "compare_airports" else {result["profile"]["airport"]: result["profile"]}
    sections = _airport_sections(profiles)
    default = list(sections) if call.name == "compare_airports" else []
    sections["passenger_growth"] = [_passenger_growth_table(profiles)]
    requested = result.get("table_sections")
    chosen = list(default if requested is None else requested)
    texts = [_table_text(sections[name]) for name in chosen]
    uncited = find_unverified_numbers(narrative, texts)
    for name in _SECTIONS_BY_SPECIFICITY:
        if name not in chosen and uncited:
            remaining = find_unverified_numbers(narrative, [*texts, _table_text(sections[name])])
            if len(remaining) < len(uncited):
                chosen.append(name)
                texts.append(_table_text(sections[name]))
                uncited = remaining
    return chosen


def focused_evidence(
    evidence: ToolEvidence, call: ToolCall, calls: list[ToolCall], narrative: str, preferred: str | None = None
) -> ToolEvidence:
    """Limit what a person sees to what this answer used. The complete tool result stays in evidence.raw."""
    if call.is_error:
        return evidence
    if call.name == "rank_opportunity_score":
        primary = primary_ranking(calls, preferred)
        led_by_another = primary is not None and primary.name != call.name
        if led_by_another and len(find_unverified_numbers(narrative, [_table_text(evidence.tables)])) == len(
            find_unverified_numbers(narrative, [])
        ):
            return replace(evidence, tables=[], warnings=[], definitions={}, provisional=[], caveats=[])
        return evidence
    if call.name not in ("compare_airports", "get_airport_profile"):
        return evidence

    chosen = _answer_sections(call, narrative)
    result = call.result
    profiles = result["airports"] if call.name == "compare_airports" else {result["profile"]["airport"]: result["profile"]}
    titles = {title for name in chosen for title in _SECTION_TITLES.get(name, ())}
    tables = [table for table in evidence.tables if table.title in titles]
    if "passenger_growth" in chosen:
        tables.insert(0, _passenger_growth_table(profiles))
    if {"demand", "passenger_growth"} & set(chosen):
        tables += [table for table in evidence.tables if table.title.endswith(" annual inputs")]

    definitions = {
        key: value
        for key, value in evidence.definitions.items()
        if any(key in _SECTION_DEFINITIONS.get(name, ()) for name in chosen)
    }
    if "congestion" in chosen:
        definitions.update(CONGESTION_DEFINITIONS)
    caveats = {caveat for name in chosen for caveat in _SECTION_CAVEATS.get(name, ())}
    sources = {tools.SOURCES[key] for name in chosen for key in _SECTION_SOURCES[name]}
    period_names = {"congestion" if name == "congestion" else "demand" for name in chosen if name != "cargo"}
    # An answer that used no table keeps its sources and periods, because they still describe the tool call.
    return replace(
        evidence,
        tables=tables,
        periods=[result["periods"][name] for name in result["periods"] if name in period_names] if chosen else evidence.periods,
        warnings=evidence.warnings if {"score", "demand", "passenger_growth"} & set(chosen) else [],
        definitions=definitions,
        sources=[source for source in evidence.sources if source in sources] if chosen else evidence.sources,
        provisional=[caveat for caveat in evidence.provisional if caveat in caveats],
        caveats=[caveat for caveat in evidence.caveats if caveat in caveats],
    )


def escape_markdown(text: str) -> str:
    """Show dollar signs literally, because Streamlit reads them as math."""
    return text.replace("$", r"\$")
