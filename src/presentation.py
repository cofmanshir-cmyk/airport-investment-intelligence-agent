"""Shape tool results for display. Values are copied, never calculated here."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import scoring
from .agent import ToolCall

PROVISIONAL_PREFIX = "PROVISIONAL"
SMALL_BASE_PREFIX = "Small base"

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
    "nas_delay_min_per_arrival": "NAS delay (min per arrival)",
    "weather_delay_min_per_arrival": "Weather delay (min per arrival)",
    "carrier_delay_min_per_arrival": "Carrier delay (min per arrival)",
    "late_aircraft_delay_min_per_arrival": "Late-aircraft delay (min per arrival)",
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
POSITION_LABEL = "Position in this list"

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
    disclaimer: str | None = None
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
        "Airport Opportunity Score",
        {code: _score_summary(section) for code, section in sections.items()},
        " ".join(["Preliminary screening proxy, not an investment recommendation.", *reasons]),
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
            "Demand-pressure proxies",
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
            f"{year_note} Separate from the demand proxies; the two are never merged.".strip(),
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
            "Airport Opportunity Score",
            pd.DataFrame(rows).reindex(columns=list(columns)).rename(columns=columns),
            _filter_note(result["filters"]),
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
    """Separate the first paragraph from the rest, so the table can sit between them.

    Results tables come only from tool output, so any table the model writes is removed.
    """
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("|")]
    parts = re.split(r"\n\s*\n", "\n".join(lines).strip(), maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def main_tables(calls: list[ToolCall]) -> list[Table]:
    """Choose the results table for each tool, from its own output."""
    tables = []
    for call in calls:
        if call.is_error:
            continue
        result = call.result
        if call.name == "rank_opportunity_score":
            tables += _opportunity_evidence(result)[0][:1] if result["ranked"] else []
        elif call.name == "rank_airports":
            tables += _rank_evidence(result)[0][:1] if result["ranked"] else []
        elif call.name in ("compare_airports", "get_airport_profile"):
            profiles = result["airports"] if call.name == "compare_airports" else {result["profile"]["airport"]: result["profile"]}
            sections = _airport_sections(profiles)
            # A comparison needs a table by default; one airport's fact often does not.
            chosen = result.get("table_sections")
            if chosen is None:
                chosen = list(sections) if call.name == "compare_airports" else []
            tables += [sections[name][0] for name in chosen if sections[name]]
    return tables


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
    evidence.disclaimer = result.get("disclaimer")
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


def escape_markdown(text: str) -> str:
    """Show dollar signs literally, because Streamlit reads them as math."""
    return text.replace("$", r"\$")
