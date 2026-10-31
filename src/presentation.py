"""Shape tool results for display. Values are copied, never calculated here."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .agent import ToolCall

PROVISIONAL_PREFIX = "PROVISIONAL"

TOOL_LABELS = {
    "rank_airports": "Passenger growth ranking",
    "compare_airports": "Airport comparison",
    "get_airport_profile": "Airport profile",
    "get_long_haul_share": "Long-haul share",
}

LABELS = {
    "passenger_cagr_3y_pct": "Passenger CAGR 3Y (%)",
    "passenger_cagr_10y_pct": "Passenger CAGR 10Y (%)",
    "flight_cagr_3y_pct": "Flight CAGR 3Y (%), all departures",
    "flight_cagr_10y_pct": "Flight CAGR 10Y (%), all departures",
    "growth_gap_3y_pts": "Growth Gap 3Y (points), all departures",
    "total_load_factor_2025_pct": "Total Load Factor 2025 (%)",
    "load_factor_change_3y_pts": "Load Factor Change 3Y (points)",
    "passengers_per_flight_growth_3y_pct": "Passengers per Flight Growth 3Y (%), all departures",
    "flight_growth_acceleration_pts": "Flight Growth Acceleration (points), all departures",
    "passenger_growth_acceleration_pts": "Passenger Growth Acceleration (points)",
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
    "passenger_cagr_pct": "Passenger CAGR (%)",
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


def _airport_tables(profiles: dict[str, dict]) -> list[Table]:
    """Keep demand, congestion and cargo in separate tables."""
    congestion = {code: profile["operational_congestion"] for code, profile in profiles.items()}
    congestion_rows = {
        code: {key: value for key, value in section.items() if key not in ("year", "months")} if section else None
        for code, section in congestion.items()
    }
    years = {section["year"] for section in congestion.values() if section}
    year_note = f"Calendar {years.pop()}." if len(years) == 1 else ""
    cargo = {code: profile["cargo_context_not_scored"] for code, profile in profiles.items()}
    tables = [
        _section_table(
            "Demand-pressure proxies",
            {code: profile["demand_pressure_proxies"]["metrics"] for code, profile in profiles.items()},
            "Raw values, unweighted and not normalized. They do not measure congestion, terminal capacity, "
            "unmet demand or profitability. Blank means a required year is incomplete.",
        ),
        _section_table(
            "Operational congestion",
            congestion_rows,
            f"{year_note} Separate from the demand proxies; the two are never merged.".strip(),
        ),
        _section_table(
            "Cargo context, not scored",
            {
                code: {"Freight (lbs)": section["freight_lbs"]} if section else None
                for code, section in cargo.items()
            },
        ),
    ]
    return [table for table in tables if table is not None]


def _annual_inputs_table(airport: str, profile: dict) -> Table:
    inputs = profile["demand_pressure_proxies"]["annual_inputs"]
    frame = pd.DataFrame(inputs).T.rename(columns=lambda name: name.replace("_", " ").capitalize())
    frame.index.name = "Year"
    return Table(f"{airport} annual inputs", frame.reset_index(), "Annual API totals behind the metrics.")


def _rank_evidence(result: dict) -> tuple[list[Table], list[str]]:
    base, end = result["base_year"], result["end_year"]
    columns = {
        **RANK_COLUMNS,
        f"passengers_{base}": f"Passengers {base}",
        f"passengers_{end}": f"Passengers {end}",
    }
    order = [*list(RANK_COLUMNS)[:-1], f"passengers_{base}", f"passengers_{end}", "passenger_cagr_pct"]
    tables = []
    if result["ranked"]:
        ranked = pd.DataFrame(result["ranked"]).reindex(columns=order).rename(columns=columns)
        tables.append(Table(
            f"Passenger CAGR {result['window_years']}Y",
            ranked,
            f"One raw metric among all {result['airports_in_national_ranking']} ranked airports, "
            "not an investment ranking.",
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
    if call.name == "rank_airports":
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
