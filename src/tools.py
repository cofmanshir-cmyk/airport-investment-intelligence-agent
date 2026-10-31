"""Tools the LLM can call. Every number returned is calculated in Python."""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import analytics, config, scoring

MAX_TOP_N = 25
MAX_COMPARE = 6

SOURCES = {
    "t100_summary": "BTS T-100 Segment Summary By Origin Airport (r495-tyji), all carriers",
    "t100_routes": "BTS T-100 Segment, US Carriers Only, Jan-Jul 2026 (manual TranStats download)",
    "faa": "FAA CY2025 commercial service enplanements",
    "ontime": "BTS On-Time Performance, 2025, reporting carriers on domestic flights",
}

NOT_A_SCORE = (
    "These are raw values, not the Airport Opportunity Score. They do not measure terminal capacity, "
    "unmet demand or profitability."
)
CARGO_INCLUSIVE = (
    f"PROVISIONAL: {scoring.CARGO_INCLUSIVE_POINTS}% of the Airport Opportunity Score "
    f"({scoring.CARGO_INCLUSIVE_POINTS} of 100 points) uses cargo-inclusive departure metrics. They count all "
    "departures, including cargo flights, and cannot be filtered in this API, so cargo activity may influence the score."
)
SIZE_BIAS = (
    "Absolute passenger growth tends to favor larger airports, so read it alongside Passenger CAGR. "
    "Neither metric replaces the other."
)
BASE_2022 = "2022, the 3-year base, was still a recovery year, which may inflate 3-year growth."
CONNECTING = "T-100 passengers include people connecting at the airport; they cannot be separated."
PASSENGER_COUNT = "T-100 airport passengers count departing passengers only, not total two-way traffic."
LOAD_FACTOR = (
    "Total Load Factor is annual passengers divided by annual seats. It is not distance-weighted "
    "and is not a passenger-only capacity measure."
)
INCOMPLETE_YEARS = (
    "A year counts only with 12 reported months and no blank totals. A month with no row cannot be confirmed "
    "as zero activity from this API, so it counts as missing. Affected metrics are null, and an airport "
    "missing any scored metric gets no score."
)
METRIC_OVERLAP = (
    "Some scored metrics overlap, notably the Growth Gap and Passengers per Departure Growth, "
    "so a single pattern can count more than once."
)
SCORE_DISCLAIMER = (
    f"Preliminary screening proxy. {scoring.CARGO_INCLUSIVE_POINTS}% of the score uses cargo-inclusive departure "
    "metrics, so cargo activity may influence it. Absolute passenger growth favors larger airports, and the 2022 "
    "base year was still in recovery, which may inflate 3-year growth. It does not measure passenger-only capacity, "
    "terminal capacity, unmet demand, profitability or investment returns."
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
    "absolute_passenger_growth_3y": "passengers 2025 minus passengers 2022",
    "flight_cagr_3y_pct": "(all departures 2025 / all departures 2022)^(1/3) - 1, percent per year, includes cargo flights",
    "flight_cagr_10y_pct": "(all departures 2025 / all departures 2015)^(1/10) - 1, percent per year, includes cargo flights",
    "growth_gap_3y_pts": "passenger CAGR 3Y minus total departure CAGR 3Y, percentage points, includes cargo flights",
    "total_load_factor_2025_pct": "annual passengers / annual seats, percent",
    "load_factor_change_3y_pts": "total load factor 2025 minus total load factor 2022, percentage points",
    "passengers_per_flight_growth_3y_pct": "3-year CAGR of passengers per departure, percent per year, includes cargo flights",
    "flight_growth_acceleration_pts": "total departure CAGR 3Y minus total departure CAGR 10Y, percentage points, includes cargo flights",
    "passenger_growth_acceleration_pts": "passenger CAGR 3Y minus passenger CAGR 10Y, percentage points",
}
SCORE_DEFINITIONS = {
    "opportunity_score": (
        "Sum of weight x normalized value / 100 across the 11 metrics. normalized = 100 x (clipped value - P5) / "
        "(P95 - P5), where the value is clipped to the 5th and 95th percentiles of the fixed national peer group."
    ),
    "national_rank": "Position among all scored airports nationally, 1 is the highest score; ties break by airport code",
}
PEER_GROUP = (
    f"Fixed national group: airports with at least {scoring.PEER_MIN_ENPLANEMENTS:,} FAA CY2025 enplanements "
    "and all 11 metrics. Geographic filters do not change it."
)


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
def _routes() -> pd.DataFrame:
    routes = _load("t100_routes")
    return routes.assign(ORIGIN=routes["ORIGIN"].replace(config.ROUTE_TO_T100_CODES))


@lru_cache(maxsize=None)
def _mix() -> pd.DataFrame:
    return analytics.route_mix(_routes())


@lru_cache(maxsize=None)
def _airports() -> pd.DataFrame:
    airports = _load("faa_airports").copy()
    airports["t100_code"] = airports["airport"].map(lambda code: config.FAA_TO_T100_CODES.get(code, code))
    return airports


@lru_cache(maxsize=None)
def _scores() -> dict:
    """Score the national peer group once. Warnings are added but never change a score."""
    result = scoring.score_airports(_annual(), _airports())
    for record in result["scored"]:
        record["warnings"] = analytics.airport_warnings(_mix(), record["airport"])
    return result


def _score_section(airport: str) -> dict:
    """Describe one airport's score, or say why it has none."""
    scores = _scores()
    inputs = analytics.year_inputs(_annual(), airport)
    metrics = analytics.raw_metrics(_annual(), airport)
    section = {
        "opportunity_score": None,
        "national_rank": None,
        "national_peer_count": scores["airports_scored"],
        "passengers_2022": inputs["2022"].get("passengers"),
        "passengers_2025": inputs["2025"].get("passengers"),
        "absolute_passenger_growth_3y": metrics["absolute_passenger_growth_3y"],
        "passenger_cagr_3y_pct": metrics["passenger_cagr_3y_pct"],
    }
    record = next((item for item in scores["scored"] if item["airport"] == airport), None)
    if record:
        section.update({
            "opportunity_score": record["opportunity_score"],
            "national_rank": record["national_rank"],
            "components": record["components"],
        })
        return section
    excluded = next((item for item in scores["not_scored"] if item["airport"] == airport), None)
    section["reason"] = (
        excluded["reason"]
        if excluded
        else f"Not in the national peer group: fewer than {scoring.PEER_MIN_ENPLANEMENTS:,} FAA CY2025 enplanements"
    )
    return section


def _selected_states(states: list[str], airports: pd.DataFrame) -> set[str]:
    selected = {state.upper() for state in states}
    unknown = sorted(selected - set(airports["state"]))
    if unknown:
        raise ToolInputError(f"Unknown state codes: {unknown}. Use 2-letter codes.")
    return selected


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
            "supplementary_not_scored": {
                "absolute_passenger_growth_10y": analytics.absolute_passenger_growth(annual, airport, 10),
            },
            "annual_inputs": analytics.year_inputs(annual, airport),
        },
        "opportunity_score": _score_section(airport),
        "operational_congestion": analytics.congestion_metrics(_load("ontime_airport_month"), annual, airport),
        "cargo_context_not_scored": cargo,
        "route_file_context": route_context,
        "warnings": [*analytics.small_base_warnings(annual, airport), *analytics.airport_warnings(mix, airport)],
    }


PROFILE_CAVEATS = [
    CARGO_INCLUSIVE, SIZE_BIAS, BASE_2022, LOAD_FACTOR, PASSENGER_COUNT, CONNECTING, INCOMPLETE_YEARS,
    METRIC_OVERLAP, THRESHOLD, CONGESTION_SCOPE,
]
PROFILE_SOURCES = [SOURCES["t100_summary"], SOURCES["ontime"], SOURCES["faa"], SOURCES["t100_routes"]]
PROFILE_DEFINITIONS = {
    **METRIC_DEFINITIONS,
    "absolute_passenger_growth_10y": "passengers 2025 minus passengers 2015, not part of the score",
    **SCORE_DEFINITIONS,
}


def get_airport_profile(airport: str) -> dict:
    code = _resolve(airport)
    return {
        "periods": {"demand": "Calendar years 2015, 2022, 2025", "congestion": "Calendar 2025"},
        "disclaimer": SCORE_DISCLAIMER,
        "profile": _profile(code),
        "peer_group": PEER_GROUP,
        "definitions": PROFILE_DEFINITIONS,
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

    def score_differences():
        scores = {code: profiles[code]["opportunity_score"]["opportunity_score"] for code in codes}
        return {
            f"{code}_minus_{codes[0]}": round(scores[code] - scores[codes[0]], 2)
            for code in codes[1:]
            if scores[code] is not None and scores[codes[0]] is not None
        }

    return {
        "periods": {"demand": "Calendar years 2015, 2022, 2025", "congestion": "Calendar 2025"},
        "disclaimer": SCORE_DISCLAIMER,
        "airports": profiles,
        "differences_from_first_airport": {
            "opportunity_score": score_differences(),
            "demand_pressure_proxies": differences("demand_pressure_proxies", "metrics"),
            "operational_congestion": differences("operational_congestion"),
        },
        "interpretation_note": (
            "Operational congestion and demand-pressure proxies are different concepts and must not be merged."
        ),
        "peer_group": PEER_GROUP,
        "definitions": PROFILE_DEFINITIONS,
        "sources": PROFILE_SOURCES,
        "caveats": PROFILE_CAVEATS,
    }


def rank_opportunity_score(states: list[str], top_n: int) -> dict:
    """Show scores in national rank order. A state filter only hides rows."""
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be between 1 and {MAX_TOP_N}.")
    selected = _selected_states(states, _airports())
    keep = lambda record: not selected or record["state"] in selected
    scores = _scores()
    matching = [record for record in scores["scored"] if keep(record)]
    ranked = [
        {
            **{key: value for key, value in record.items() if key != "components"},
            "score_points": {metric: part["points"] for metric, part in record["components"].items()},
        }
        for record in matching[:top_n]
    ]
    return {
        "metric": "Airport Opportunity Score",
        "disclaimer": SCORE_DISCLAIMER,
        "periods": "Full calendar years 2015, 2022 and 2025",
        "filters": {"states": sorted(selected) or "all"},
        "peer_group": PEER_GROUP,
        "airports_eligible": scores["airports_eligible"],
        "airports_scored_nationally": scores["airports_scored"],
        "airports_matching_filters": len(matching),
        "ranked": ranked,
        "not_scored": [record for record in scores["not_scored"] if keep(record)],
        "weights": scoring.WEIGHTS,
        "normalization_limits": scores["bounds"],
        "definitions": {**METRIC_DEFINITIONS, **SCORE_DEFINITIONS},
        "sources": [SOURCES["t100_summary"], SOURCES["faa"], SOURCES["t100_routes"]],
        "caveats": [
            CARGO_INCLUSIVE, SIZE_BIAS, BASE_2022, LOAD_FACTOR, PASSENGER_COUNT, CONNECTING, INCOMPLETE_YEARS,
            METRIC_OVERLAP, THRESHOLD,
            "Small-base warnings and route-file warnings never change a score or rank.",
        ],
    }


def rank_airports(
    window_years: int, states: list[str], top_n: int, min_enplanements: int, rank_by: str = "passenger_cagr"
) -> dict:
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be between 1 and {MAX_TOP_N}.")
    if window_years not in analytics.BASE_YEARS:
        raise ToolInputError("window_years must be 3 or 10.")
    if min_enplanements not in (50_000, 100_000, 250_000):
        raise ToolInputError("min_enplanements must be 50000, 100000 or 250000.")
    if rank_by not in analytics.RANK_METRICS:
        raise ToolInputError(f"rank_by must be one of {sorted(analytics.RANK_METRICS)}.")

    airports = _airports()
    _selected_states(states, airports)

    result = analytics.rank_passenger_growth(
        _annual(), airports, _mix(), window_years, min_enplanements, states, rank_by
    )
    matching = len(result["ranked"])
    result["ranked"] = result["ranked"][:top_n]
    absolute = rank_by == "absolute_passenger_growth"
    name = "Absolute Passenger Growth" if absolute else "Passenger CAGR"
    role = (
        "this is not part of the score"
        if absolute and window_years == 10
        else "this is one input to the score, not the score"
    )
    return {
        "metric": f"{name} {window_years}Y: {role}",
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
            "absolute_passenger_growth": f"passengers {result['end_year']} minus passengers {result['base_year']}",
            "national_rank": (
                "Position among all eligible airports nationally, 1 is the largest absolute increase"
                if absolute
                else "Position among all eligible airports nationally, 1 is fastest growth"
            ),
        },
        "sources": [SOURCES["t100_summary"], SOURCES["faa"], SOURCES["t100_routes"]],
        "caveats": [
            NOT_A_SCORE, SIZE_BIAS, PASSENGER_COUNT, THRESHOLD, INCOMPLETE_YEARS, CONNECTING,
            *([BASE_2022] if window_years == 3 else []),
            "PROVISIONAL: warning thresholds only add flags and never change a value or rank.",
            "Passenger totals include all carriers and service classes; the class F versus F+L decision is open.",
        ],
    }


def get_long_haul_share(airport: str, threshold_miles: int) -> dict:
    code = _resolve(airport)
    try:
        result = analytics.long_haul_share(
            _routes(), code, threshold_miles, airport_summary=_load("t100_airport_month")
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
        "name": "rank_opportunity_score",
        "description": (
            "Rank US airports by the Airport Opportunity Score, a preliminary screening proxy built from 11 weighted "
            "metrics. Use when the user asks about the score or a score ranking. Scores use a fixed national peer "
            "group, so a state filter only limits which airports are shown. Pass a list of states for a region; "
            "New England is CT, MA, ME, NH, RI, VT."
        ),
        "strict": True,
        "input_schema": _schema({
            "states": {"type": "array", "items": {"type": "string"}, "description": "2-letter state codes; empty for all"},
            "top_n": {"type": "integer", "description": f"Airports to return, 1 to {MAX_TOP_N}"},
        }),
    },
    {
        "name": "rank_airports",
        "description": (
            "Rank US airports on one passenger growth metric, either CAGR or absolute growth. Use for fastest-growing "
            "airports, the most added passengers, or passenger growth in a state or region. Pass a list of states for "
            "a region; New England is CT, MA, ME, NH, RI, VT. This is not the Airport Opportunity Score."
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
            "rank_by": {
                "type": "string", "enum": sorted(analytics.RANK_METRICS),
                "description": (
                    "passenger_cagr for percentage growth; absolute_passenger_growth for passengers added. "
                    "Use passenger_cagr unless the user asks for the number of passengers added"
                ),
            },
        }),
    },
    {
        "name": "compare_airports",
        "description": (
            "Compare 2 to 6 airports on the Airport Opportunity Score, its 11 metrics, operational congestion "
            "(delays, taxi-out, delay causes) and cargo context. Use for comparisons, including congestion questions."
        ),
        "strict": True,
        "input_schema": _schema({
            "airports": {"type": "array", "items": AIRPORT_CODE, "description": "Airport codes; the first is the reference"},
        }),
    },
    {
        "name": "get_airport_profile",
        "description": (
            "Get one airport's Airport Opportunity Score, its 11 metrics, annual inputs, operational congestion, "
            "cargo context and warnings. Use for questions about one airport."
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
    "rank_opportunity_score": rank_opportunity_score,
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
