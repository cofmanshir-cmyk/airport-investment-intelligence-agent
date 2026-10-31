"""Airport Opportunity Score: a preliminary screening proxy, calculated only in Python.

Every airport is treated the same way. The peer group comes from the data, and
this module selects no airport by name.
"""

from __future__ import annotations

import pandas as pd

from . import analytics, config

# Fixed weights in points. They total 100.
WEIGHTS = {
    "passenger_cagr_3y_pct": 15,
    "passenger_cagr_10y_pct": 10,
    "absolute_passenger_growth_3y": 10,
    "flight_cagr_3y_pct": 5,
    "flight_cagr_10y_pct": 5,
    "growth_gap_3y_pts": 15,
    "total_load_factor_2025_pct": 15,
    "load_factor_change_3y_pts": 10,
    "passengers_per_flight_growth_3y_pct": 5,
    "flight_growth_acceleration_pts": 5,
    "passenger_growth_acceleration_pts": 5,
}
# These metrics use all departures, including cargo flights.
CARGO_INCLUSIVE_METRICS = (
    "flight_cagr_3y_pct",
    "flight_cagr_10y_pct",
    "growth_gap_3y_pts",
    "passengers_per_flight_growth_3y_pct",
    "flight_growth_acceleration_pts",
)
CARGO_INCLUSIVE_POINTS = sum(WEIGHTS[metric] for metric in CARGO_INCLUSIVE_METRICS)
LOWER_PERCENTILE = 0.05
UPPER_PERCENTILE = 0.95
PEER_MIN_ENPLANEMENTS = config.DEFAULT_MIN_ENPLANEMENTS


def normalization_bounds(values: pd.Series) -> tuple[float, float]:
    """Return the peer group's 5th and 95th percentiles, using linear interpolation."""
    lower, upper = values.quantile([LOWER_PERCENTILE, UPPER_PERCENTILE], interpolation="linear")
    if upper == lower:
        raise ValueError(f"Cannot normalize {values.name}: its percentile limits are equal")
    return float(lower), float(upper)


def normalize(value: float, lower: float, upper: float) -> float:
    """Clip a value to its limits and scale it linearly from 0 to 100."""
    return 100 * ((min(max(value, lower), upper) - lower) / (upper - lower))


def _not_scored_reason(annual: pd.DataFrame, airport: str) -> str:
    for year in config.HISTORY_YEARS:
        if analytics.complete_year(annual, airport, year) is None:
            months = int(annual.loc[(airport, year), "months"]) if (airport, year) in annual.index else 0
            if months < 12:
                return f"{year} is incomplete: {months} of 12 months reported"
            return f"{year} has blank or zero annual totals"
    return "A required metric is unavailable"


def _identity(row) -> dict:
    return {
        "airport": row.t100_code,
        "airport_name": row.airport_name,
        "state": row.state,
        "faa_hub_class": row.hub if isinstance(row.hub, str) else "none",
        "faa_enplanements_2025": int(row.enplanements_2025),
    }


def score_airports(annual: pd.DataFrame, airports: pd.DataFrame) -> dict:
    """Score every eligible airport against the fixed national peer group.

    Peers are eligible airports with all 11 metrics. Limits are calculated once, from
    unrounded values and before any filter, so a geographic filter cannot change a score.
    An airport missing a metric gets no score, and the other weights are never adjusted.
    """
    eligible = airports[airports["enplanements_2025"] >= PEER_MIN_ENPLANEMENTS].sort_values("t100_code")
    values, identities, not_scored = {}, {}, []
    for row in eligible.itertuples():
        metrics = analytics.metric_values(annual, row.t100_code)
        identities[row.t100_code] = _identity(row)
        if any(metrics[metric] is None for metric in WEIGHTS):
            not_scored.append({**_identity(row), "reason": _not_scored_reason(annual, row.t100_code)})
        else:
            values[row.t100_code] = metrics

    peers = pd.DataFrame.from_dict(values, orient="index", columns=list(WEIGHTS)).astype(float)
    bounds = {metric: normalization_bounds(peers[metric]) for metric in WEIGHTS}

    scored = []
    for airport, metrics in values.items():
        normalized = {metric: normalize(metrics[metric], *bounds[metric]) for metric in WEIGHTS}
        score = sum(WEIGHTS[metric] * normalized[metric] for metric in WEIGHTS) / 100
        scored.append((score, airport, metrics, normalized))
    scored.sort(key=lambda item: (-item[0], item[1]))

    records = []
    for rank, (score, airport, metrics, normalized) in enumerate(scored, start=1):
        rounded = analytics.raw_metrics(annual, airport)
        inputs = analytics.year_inputs(annual, airport)
        records.append({
            "national_rank": rank,
            **identities[airport],
            "opportunity_score": round(score, 2),
            f"passengers_{config.BASE_YEAR_3Y}": inputs[str(config.BASE_YEAR_3Y)]["passengers"],
            f"passengers_{config.LATEST_COMPLETE_YEAR}": inputs[str(config.LATEST_COMPLETE_YEAR)]["passengers"],
            "absolute_passenger_growth_3y": rounded["absolute_passenger_growth_3y"],
            "passenger_cagr_3y_pct": rounded["passenger_cagr_3y_pct"],
            "components": {
                metric: {
                    "raw_value": rounded[metric],
                    "normalized": round(normalized[metric], 2),
                    "weight": WEIGHTS[metric],
                    "points": round(WEIGHTS[metric] * normalized[metric] / 100, 2),
                }
                for metric in WEIGHTS
            },
            "small_base_warnings": analytics.small_base_warnings(annual, airport),
        })

    return {
        "airports_eligible": len(eligible),
        "airports_scored": len(records),
        "scored": records,
        "not_scored": not_scored,
        "bounds": {metric: {"lower": lower, "upper": upper} for metric, (lower, upper) in bounds.items()},
    }
