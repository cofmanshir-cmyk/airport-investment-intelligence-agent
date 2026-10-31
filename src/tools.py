"""Tools the LLM can call. Every number returned is calculated in Python."""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import analytics, config

MAX_TOP_N = 25
MAX_COMPARE = 6

SOURCES = {
    "t100_summary": "BTS T-100 Segment Summary By Origin Airport (r495-tyji), all carriers",
    "t100_routes": "BTS T-100 Segment, US Carriers Only, Jan-Jul 2026 (manual TranStats download)",
    "faa": "FAA CY2025 commercial service enplanements",
    "ontime": "BTS On-Time Performance, 2025, reporting carriers on domestic flights",
}

NOT_A_SCORE = (
    "These are raw values. They are not the Airport Opportunity Score, which is not implemented, "
    "and they do not measure terminal capacity, unmet demand or profitability."
)
FLIGHTS_PROVISIONAL = (
    "PROVISIONAL: flight metrics use all API departures, including freighters. The agreed "
    "passenger-only definition needs historical T-100 All Carriers route data, not yet loaded."
)
BASE_2022 = "2022, the 3-year base, was still a recovery year, which inflates 3-year growth."
CONNECTING = "T-100 passengers include people connecting at the airport; they cannot be separated."
INCOMPLETE_YEARS = (
    "PROVISIONAL: a year counts only with 12 reported months. How to treat incomplete years "
    "is an open decision, so affected values are shown as null."
)
THRESHOLD = (
    "PROVISIONAL: the eligibility threshold uses FAA CY2025 enplanements and is a filter, "
    "not a scoring metric."
)
CONGESTION_SCOPE = (
    "Operational congestion comes from reporting carriers on domestic flights only, so international "
    "and most cargo flights are absent. Delay-cause minutes attach to arriving flights and may "
    "originate elsewhere in the network."
)

METRIC_DEFINITIONS = {
    "passenger_cagr_3y_pct": "(passengers 2025 / passengers 2022)^(1/3) - 1, percent per year",
    "passenger_cagr_10y_pct": "(passengers 2025 / passengers 2015)^(1/10) - 1, percent per year",
    "flight_cagr_3y_pct": "(departures 2025 / departures 2022)^(1/3) - 1, percent per year",
    "flight_cagr_10y_pct": "(departures 2025 / departures 2015)^(1/10) - 1, percent per year",
    "growth_gap_3y_pts": "passenger CAGR 3Y minus flight CAGR 3Y, percentage points",
    "total_load_factor_2025_pct": "annual passengers / annual seats, percent",
    "load_factor_change_3y_pts": "load factor 2025 minus load factor 2022, percentage points",
    "passengers_per_flight_growth_3y_pct": "3-year CAGR of passengers per departure, percent per year",
    "flight_growth_acceleration_pts": "flight CAGR 3Y minus flight CAGR 10Y, percentage points",
    "passenger_growth_acceleration_pts": "passenger CAGR 3Y minus passenger CAGR 10Y, percentage points",
}


class ToolInputError(ValueError):
    """A tool input the LLM can correct, returned to it as a tool error."""


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run python -m src.pipeline")
    return pd.read_parquet(path)


@lru_cache(maxsize=None)
def _annual() -> pd.DataFrame:
    return analytics.annual_totals(_load("t100_airport_month"))


@lru_cache(maxsize=None)
def _mix() -> pd.DataFrame:
    return analytics.route_mix(_load("t100_routes"))


@lru_cache(maxsize=None)
def _airports() -> pd.DataFrame:
    airports = _load("faa_airports").copy()
    airports["t100_code"] = airports["airport"].map(lambda code: config.FAA_TO_T100_CODES.get(code, code))
    return airports


def _resolve(code: str) -> str:
    """Accept a T-100 or FAA airport code and return the T-100 code."""
    cleaned = str(code).strip().upper()
    cleaned = config.FAA_TO_T100_CODES.get(cleaned, cleaned)
    if cleaned not in _annual().index.get_level_values("airport"):
        raise ToolInputError(f"Airport code {code!r} is not in the T-100 data. Use a 3-letter code.")
    return cleaned


def _profile(airport: str) -> dict:
    annual = _annual()
    airports = _airports()
    faa = airports[airports["t100_code"] == airport]
    identity = {"airport": airport, "in_faa_commercial_service_list": not faa.empty}
    if not faa.empty:
        row = faa.iloc[0]
        identity.update({
            "airport_name": row["airport_name"],
            "city": row["city"],
            "state": row["state"],
            "faa_hub_class": row["hub"] if isinstance(row["hub"], str) else "none",
            "faa_enplanements_2025": int(row["enplanements_2025"]),
        })

    cargo = None
    if (airport, analytics.END_YEAR) in annual.index:
        cargo = {"year": analytics.END_YEAR, "freight_lbs": int(annual.loc[(airport, analytics.END_YEAR), "freight_lbs"])}

    mix = _mix()
    route_context = None
    if airport in mix.index:
        route_context = {
            "period": "Jan-Jul 2026, US carriers only",
            "passenger_aircraft_share_of_departures_pct": float(mix.loc[airport, "passenger_aircraft_share_pct"]),
            "scheduled_share_of_passengers_pct": float(mix.loc[airport, "scheduled_passenger_share_pct"]),
        }

    return {
        **identity,
        "demand_pressure_proxies": {
            "metrics": analytics.raw_metrics(annual, airport),
            "annual_inputs": analytics.year_inputs(annual, airport),
        },
        "operational_congestion": analytics.congestion_metrics(_load("ontime_airport_month"), annual, airport),
        "cargo_context_not_scored": cargo,
        "route_file_context": route_context,
        "warnings": analytics.airport_warnings(mix, airport),
    }


PROFILE_CAVEATS = [NOT_A_SCORE, FLIGHTS_PROVISIONAL, BASE_2022, CONNECTING, INCOMPLETE_YEARS, CONGESTION_SCOPE]
PROFILE_SOURCES = [SOURCES["t100_summary"], SOURCES["ontime"], SOURCES["faa"], SOURCES["t100_routes"]]


def get_airport_profile(airport: str) -> dict:
    code = _resolve(airport)
    return {
        "periods": {"demand": "Calendar years 2015, 2022, 2025", "congestion": "Calendar 2025"},
        "profile": _profile(code),
        "definitions": METRIC_DEFINITIONS,
        "sources": PROFILE_SOURCES,
        "caveats": PROFILE_CAVEATS,
    }


def compare_airports(airports: list[str]) -> dict:
    if not 2 <= len(airports) <= MAX_COMPARE:
        raise ToolInputError(f"Provide between 2 and {MAX_COMPARE} airport codes.")
    codes = [_resolve(code) for code in airports]
    if len(set(codes)) != len(codes):
        raise ToolInputError("Airport codes must be different.")
    profiles = {code: _profile(code) for code in codes}

    def differences(section, *path):
        """Calculate each measure's difference from the first airport."""
        def values(code):
            node = profiles[code][section]
            for key in path:
                node = None if node is None else node.get(key)
            return node or {}

        first = values(codes[0])
        table = {}
        for code in codes[1:]:
            table[f"{code}_minus_{codes[0]}"] = {
                key: round(values(code)[key] - value, 2)
                for key, value in first.items()
                if isinstance(value, (int, float)) and isinstance(values(code).get(key), (int, float))
                and key not in ("year", "months")
            }
        return table

    return {
        "periods": {"demand": "Calendar years 2015, 2022, 2025", "congestion": "Calendar 2025"},
        "airports": profiles,
        "differences_from_first_airport": {
            "demand_pressure_proxies": differences("demand_pressure_proxies", "metrics"),
            "operational_congestion": differences("operational_congestion"),
        },
        "interpretation_note": (
            "Operational congestion and demand-pressure proxies are different concepts and must not be merged."
        ),
        "definitions": METRIC_DEFINITIONS,
        "sources": PROFILE_SOURCES,
        "caveats": PROFILE_CAVEATS,
    }


def rank_airports(window_years: int, states: list[str], top_n: int, min_enplanements: int) -> dict:
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be between 1 and {MAX_TOP_N}.")
    if window_years not in analytics.BASE_YEARS:
        raise ToolInputError("window_years must be 3 or 10.")
    if min_enplanements not in (50_000, 100_000, 250_000):
        raise ToolInputError("min_enplanements must be 50000, 100000 or 250000.")

    airports = _airports()
    unknown = sorted({state.upper() for state in states} - set(airports["state"]))
    if unknown:
        raise ToolInputError(f"Unknown state codes: {unknown}. Use 2-letter codes.")

    result = analytics.rank_passenger_growth(_annual(), airports, _mix(), window_years, min_enplanements, states)
    matching = len(result["ranked"])
    result["ranked"] = result["ranked"][:top_n]
    return {
        "metric": f"Passenger CAGR {window_years}Y: this is one input to the score, not the score",
        "periods": f"Full calendar years {result['base_year']} and {result['end_year']}",
        "filters": {
            "min_faa_enplanements_2025": min_enplanements,
            "states": [state.upper() for state in states] or "all",
        },
        "airports_matching_filters": matching,
        **result,
        "definitions": {
            "passenger_cagr_pct": (
                f"(passengers {result['end_year']} / passengers {result['base_year']})"
                f"^(1/{window_years}) - 1, percent per year"
            ),
            "national_rank": "Position among all eligible airports nationally, 1 is fastest growth",
        },
        "sources": [SOURCES["t100_summary"], SOURCES["faa"], SOURCES["t100_routes"]],
        "caveats": [
            NOT_A_SCORE, THRESHOLD, INCOMPLETE_YEARS, CONNECTING,
            *([BASE_2022] if window_years == 3 else []),
            "PROVISIONAL: warning thresholds only add flags and never change a value or rank.",
            "Passenger totals include all carriers and service classes; the class F versus F+L decision is open.",
        ],
    }


def get_long_haul_share(airport: str, threshold_miles: int) -> dict:
    code = _resolve(airport)
    try:
        result = analytics.long_haul_share(
            _load("t100_routes"), code, threshold_miles, airport_summary=_load("t100_airport_month")
        )
    except ValueError as error:
        raise ToolInputError(str(error)) from error
    return {
        **result,
        "definitions": {
            "share_pct": "Departures at or beyond the threshold / all departures in scope, weighted by DEPARTURES_PERFORMED",
            "passenger": "AIRCRAFT_CONFIG 1 (passenger configuration)",
            "scheduled_passenger": "AIRCRAFT_CONFIG 1 and CLASS F (scheduled passenger/cargo service)",
            "freighter": "AIRCRAFT_CONFIG 2 (freight configuration)",
        },
        "sources": [SOURCES["t100_routes"], SOURCES["t100_summary"]],
        "caveats": [
            "The threshold is an assumption; compare the sensitivity values before relying on one figure.",
            "Bounds assume the departures missing from the US-carrier file are all short-haul or all long-haul.",
        ],
    }


def _schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


AIRPORT_CODE = {"type": "string", "description": "3-letter airport code, for example SFO"}

TOOL_DEFINITIONS = [
    {
        "name": "rank_airports",
        "description": (
            "Rank US airports by raw passenger CAGR. Use for fastest-growing airports or passenger growth "
            "in a state or region. Pass a list of states for a region; New England is CT, MA, ME, NH, RI, VT. "
            "Ranks one metric only, not an investment score."
        ),
        "strict": True,
        "input_schema": _schema({
            "window_years": {"type": "integer", "enum": [3, 10], "description": "3 for 2022-2025, 10 for 2015-2025"},
            "states": {"type": "array", "items": {"type": "string"}, "description": "2-letter state codes; empty for all"},
            "top_n": {"type": "integer", "description": f"Airports to return, 1 to {MAX_TOP_N}"},
            "min_enplanements": {
                "type": "integer", "enum": [50000, 100000, 250000],
                "description": "Eligibility threshold; use 100000 unless the user asks otherwise",
            },
        }),
    },
    {
        "name": "compare_airports",
        "description": (
            "Compare 2 to 6 airports on the ten raw demand metrics, operational congestion (delays, taxi-out, "
            "delay causes) and cargo context. Use for comparisons, including congestion questions."
        ),
        "strict": True,
        "input_schema": _schema({
            "airports": {"type": "array", "items": AIRPORT_CODE, "description": "Airport codes; the first is the reference"},
        }),
    },
    {
        "name": "get_airport_profile",
        "description": (
            "Get one airport's ten raw demand metrics, annual inputs, operational congestion, cargo context "
            "and warnings. Use for questions about one airport."
        ),
        "strict": True,
        "input_schema": _schema({"airport": AIRPORT_CODE}),
    },
    {
        "name": "get_long_haul_share",
        "description": (
            "Get the share of an airport's departures at or beyond a distance, weighted by departures performed, "
            "for all flights, passenger aircraft and freighters, with coverage limits. Use for long-haul questions."
        ),
        "strict": True,
        "input_schema": _schema({
            "airport": AIRPORT_CODE,
            "threshold_miles": {
                "type": "integer",
                "description": f"Statute miles; use {config.DEFAULT_LONG_HAUL_MILES} unless the user names a distance",
            },
        }),
    },
]

_TOOLS = {
    "rank_airports": rank_airports,
    "compare_airports": compare_airports,
    "get_airport_profile": get_airport_profile,
    "get_long_haul_share": get_long_haul_share,
}


def execute_tool(name: str, tool_input: dict) -> tuple[dict, bool]:
    """Run a tool and return its result and whether it failed.

    Only input problems the LLM can correct become tool errors; anything else,
    such as missing data files, is a genuine failure and is raised.
    """
    if name not in _TOOLS:
        return {"error": f"Unknown tool {name!r}."}, True
    try:
        return _TOOLS[name](**tool_input), False
    except (ToolInputError, TypeError) as error:
        return {"error": str(error)}, True


def to_json(result: dict) -> str:
    return json.dumps(result, default=float, ensure_ascii=False)
