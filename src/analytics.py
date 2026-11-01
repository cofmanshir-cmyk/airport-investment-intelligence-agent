"""Deterministic airport analytics. No LLM calls. Scoring lives in scoring.py."""

from __future__ import annotations

import pandas as pd

from . import config

SCOPES = ("all", "passenger", "scheduled_passenger", "freighter")

END_YEAR = config.LATEST_COMPLETE_YEAR
BASE_YEARS = {3: config.BASE_YEAR_3Y, 10: config.BASE_YEAR_10Y}
RANK_METRICS = {
    "passenger_cagr": "passenger_cagr_pct",
    "absolute_passenger_growth": "absolute_passenger_growth",
}


def annual_totals(summary: pd.DataFrame) -> pd.DataFrame:
    """Sum the airport-month summary to one row per airport and calendar year.

    A blank total is missing data, so it is counted here instead of being summed as zero.
    """
    measures = summary[["passengers", "departures", "seats"]]
    return summary.assign(blank_month=measures.isna().any(axis=1)).groupby(["airport", "year"]).agg(
        passengers=("passengers", "sum"),
        departures=("departures", "sum"),
        seats=("seats", "sum"),
        freight_lbs=("freight_lbs", "sum"),
        domestic_departures=("domestic_departures", "sum"),
        months=("reporting_month", "nunique"),
        blank_months=("blank_month", "sum"),
    )


def complete_year(annual: pd.DataFrame, airport: str, year: int) -> pd.Series | None:
    """Return a year's totals only with 12 reported months, no blanks and positive activity.

    A reported month with zero passengers is a genuine zero. A month with no row
    cannot be confirmed as zero activity from this source, so it counts as missing.
    """
    if (airport, year) not in annual.index:
        return None
    row = annual.loc[(airport, year)]
    usable = (
        row["months"] == 12
        and row["blank_months"] == 0
        and row["passengers"] > 0
        and row["departures"] > 0
        and row["seats"] > 0
    )
    return row if usable else None


def cagr_pct(end: float, start: float, years: int) -> float:
    return ((end / start) ** (1 / years) - 1) * 100


def absolute_passenger_growth(annual: pd.DataFrame, airport: str, window_years: int) -> int | None:
    """Passengers in the end year minus the base year, or None if either is incomplete."""
    end = complete_year(annual, airport, END_YEAR)
    base = complete_year(annual, airport, BASE_YEARS[window_years])
    return None if end is None or base is None else int(end["passengers"]) - int(base["passengers"])


def raw_metrics(annual: pd.DataFrame, airport: str) -> dict:
    """Report the 11 scored metrics for display, rounded and not normalized."""
    return {
        name: value if value is None or isinstance(value, int) else round(float(value), 2)
        for name, value in metric_values(annual, airport).items()
    }


def metric_values(annual: pd.DataFrame, airport: str) -> dict:
    """Calculate the 11 scored metrics as unrounded raw values.

    A metric is None when a year it needs is incomplete; nothing is imputed.
    """
    rows = {year: complete_year(annual, airport, year) for year in config.HISTORY_YEARS}
    end, base_3y, base_10y = rows[END_YEAR], rows[BASE_YEARS[3]], rows[BASE_YEARS[10]]

    def growth(column, base, window):
        if end is None or base is None:
            return None
        return cagr_pct(end[column], base[column], window)

    def difference(a, b):
        return None if a is None or b is None else a - b

    def load_factor(row):
        return None if row is None else 100 * row["passengers"] / row["seats"]

    passengers_3y, passengers_10y = growth("passengers", base_3y, 3), growth("passengers", base_10y, 10)
    flights_3y, flights_10y = growth("departures", base_3y, 3), growth("departures", base_10y, 10)
    per_flight_3y = None
    if end is not None and base_3y is not None:
        per_flight_3y = cagr_pct(
            end["passengers"] / end["departures"], base_3y["passengers"] / base_3y["departures"], 3
        )

    return {
        "passenger_cagr_3y_pct": passengers_3y,
        "passenger_cagr_10y_pct": passengers_10y,
        "absolute_passenger_growth_3y": absolute_passenger_growth(annual, airport, 3),
        "flight_cagr_3y_pct": flights_3y,
        "flight_cagr_10y_pct": flights_10y,
        "growth_gap_3y_pts": difference(passengers_3y, flights_3y),
        "total_load_factor_2025_pct": load_factor(end),
        "load_factor_change_3y_pts": difference(load_factor(end), load_factor(base_3y)),
        "passengers_per_flight_growth_3y_pct": per_flight_3y,
        "flight_growth_acceleration_pts": difference(flights_3y, flights_10y),
        "passenger_growth_acceleration_pts": difference(passengers_3y, passengers_10y),
    }


def year_inputs(annual: pd.DataFrame, airport: str) -> dict:
    """Report the annual inputs behind the metrics, including incomplete years."""
    inputs = {}
    for year in config.HISTORY_YEARS:
        if (airport, year) not in annual.index:
            inputs[str(year)] = {"months_reported": 0, "complete": False}
            continue
        row = annual.loc[(airport, year)]
        inputs[str(year)] = {
            "passengers": int(row["passengers"]),
            "departures": int(row["departures"]),
            "seats": int(row["seats"]),
            "months_reported": int(row["months"]),
            "complete": complete_year(annual, airport, year) is not None,
        }
    return inputs


def congestion_metrics(ontime: pd.DataFrame, annual: pd.DataFrame, airport: str) -> dict | None:
    """Report operational congestion for display, rounded to one decimal."""
    values = congestion_values(ontime, annual, airport)
    if values is None:
        return None
    return {name: round(value, 1) if isinstance(value, float) else value for name, value in values.items()}


def congestion_values(ontime: pd.DataFrame, annual: pd.DataFrame, airport: str) -> dict | None:
    """Calculate operational congestion for a year of reporting-carrier flights, unrounded."""
    rows = ontime[ontime["airport"] == airport]
    if rows.empty:
        return None
    totals = rows[[column for column in rows.columns if column not in ("airport", "year", "month")]].sum()

    def ratio(top, bottom, scale=1.0):
        return float(scale * totals[top] / totals[bottom]) if totals[bottom] > 0 else None

    t100_departures = None
    year = int(rows["year"].iloc[0])
    if (airport, year) in annual.index:
        t100_departures = float(annual.loc[(airport, year), "domestic_departures"])
    return {
        "year": year,
        "months": int(rows["month"].nunique()),
        "departures_in_sample": int(totals["departures"]),
        "share_of_t100_domestic_departures_pct": (
            100 * float(totals["departures"]) / t100_departures if t100_departures else None
        ),
        "departures_delayed_15_plus_min_pct": ratio("dep_del15", "dep_del15_reported", 100),
        "cancelled_departures_pct": ratio("cancelled", "departures", 100),
        "average_taxi_out_min": ratio("taxi_out_min", "taxi_out_reported"),
        "arrivals_delayed_15_plus_min_pct": ratio("arr_del15", "arr_del15_reported", 100),
        "nas_delay_min_per_arrival": ratio("nas_delay_min", "arrivals"),
        "weather_delay_min_per_arrival": ratio("weather_delay_min", "arrivals"),
        "carrier_delay_min_per_arrival": ratio("carrier_delay_min", "arrivals"),
        "late_aircraft_delay_min_per_arrival": ratio("late_aircraft_delay_min", "arrivals"),
    }


def route_mix(routes: pd.DataFrame) -> pd.DataFrame:
    """Measure each origin's aircraft and service mix in the route file."""
    flown = routes[routes["DEPARTURES_PERFORMED"] > 0]
    passenger = flown[flown["AIRCRAFT_CONFIG"] == config.CONFIG_PASSENGER]
    scheduled = passenger[passenger["CLASS"] == config.CLASS_SCHEDULED]
    mix = pd.DataFrame({
        "departures": flown.groupby("ORIGIN")["DEPARTURES_PERFORMED"].sum(),
        "passengers": flown.groupby("ORIGIN")["PASSENGERS"].sum(),
        "passenger_aircraft_departures": passenger.groupby("ORIGIN")["DEPARTURES_PERFORMED"].sum(),
        "scheduled_passengers": scheduled.groupby("ORIGIN")["PASSENGERS"].sum(),
    }).fillna(0)
    mix["passenger_aircraft_share_pct"] = (100 * mix["passenger_aircraft_departures"] / mix["departures"]).round(1)
    mix["scheduled_passenger_share_pct"] = (
        (100 * mix["scheduled_passengers"] / mix["passengers"].where(mix["passengers"] > 0)).round(1)
    )
    return mix


def _small_base_warning(year: int, passengers: float) -> str:
    return (
        f"Small base year: {year} had {int(passengers):,} passengers, below the "
        f"{config.SMALL_BASE_WARNING_PASSENGERS:,} threshold, so percentage growth may be inflated."
    )


def small_base_warnings(annual: pd.DataFrame, airport: str) -> list[str]:
    """Flag every growth base year with few passengers. Flags never change a value or rank."""
    warnings = []
    for year in sorted(BASE_YEARS.values()):
        if (airport, year) in annual.index:
            passengers = annual.loc[(airport, year), "passengers"]
            if passengers < config.SMALL_BASE_WARNING_PASSENGERS:
                warnings.append(_small_base_warning(year, passengers))
    return warnings


def airport_warnings(
    mix: pd.DataFrame, airport: str, base_passengers: float | None = None, base_year: int | None = None
) -> list[str]:
    """Describe data conditions that can mislead, without changing any value."""
    warnings = []
    if base_passengers is not None and base_passengers < config.SMALL_BASE_WARNING_PASSENGERS:
        warnings.append(_small_base_warning(base_year, base_passengers))
    if airport in mix.index:
        row = mix.loc[airport]
        scheduled = row["scheduled_passenger_share_pct"]
        if pd.notna(scheduled) and scheduled < config.SCHEDULED_PASSENGER_WARNING_PCT:
            warnings.append(
                f"Only {scheduled:g}% of its Jan-Jul 2026 US-carrier passengers flew on scheduled "
                "passenger aircraft (class F), so non-scheduled traffic may drive these figures."
            )
        if row["passenger_aircraft_share_pct"] < config.PASSENGER_AIRCRAFT_WARNING_PCT:
            warnings.append(
                f"Only {row['passenger_aircraft_share_pct']:g}% of its Jan-Jul 2026 US-carrier departures "
                "used passenger aircraft, so departure metrics are distorted by freighters."
            )
    return warnings


def rank_passenger_growth(
    annual: pd.DataFrame,
    airports: pd.DataFrame,
    mix: pd.DataFrame,
    window_years: int,
    min_enplanements: int,
    states: list[str] | None = None,
    rank_by: str = "passenger_cagr",
) -> dict:
    """Rank eligible airports by passenger CAGR or absolute growth, with ties broken by code.

    National rank is set before any state filter, so a regional list still shows
    each airport's position in the full eligible group.
    """
    if window_years not in BASE_YEARS:
        raise ValueError(f"window_years must be one of {sorted(BASE_YEARS)}")
    if rank_by not in RANK_METRICS:
        raise ValueError(f"rank_by must be one of {sorted(RANK_METRICS)}")
    base_year = BASE_YEARS[window_years]
    eligible = airports[airports["enplanements_2025"] >= min_enplanements]

    ranked, not_ranked = [], []
    for row in eligible.itertuples():
        base = complete_year(annual, row.t100_code, base_year)
        end = complete_year(annual, row.t100_code, END_YEAR)
        record = {
            "airport": row.t100_code,
            "airport_name": row.airport_name,
            "state": row.state,
            "faa_hub_class": row.hub if isinstance(row.hub, str) else "none",
            "faa_enplanements_2025": int(row.enplanements_2025),
        }
        if base is None or end is None:
            missing = base_year if base is None else END_YEAR
            months = 0
            if (row.t100_code, missing) in annual.index:
                months = int(annual.loc[(row.t100_code, missing), "months"])
            not_ranked.append({**record, "reason": f"{missing} is incomplete: {months} of 12 months reported"})
            continue
        record.update({
            f"passengers_{base_year}": int(base["passengers"]),
            f"passengers_{END_YEAR}": int(end["passengers"]),
            "absolute_passenger_growth": int(end["passengers"]) - int(base["passengers"]),
            "passenger_cagr_pct": round(float(cagr_pct(end["passengers"], base["passengers"], window_years)), 2),
            "warnings": airport_warnings(mix, row.t100_code, float(base["passengers"]), base_year),
        })
        ranked.append(record)

    ranked.sort(key=lambda item: (-item[RANK_METRICS[rank_by]], item["airport"]))
    for position, record in enumerate(ranked, start=1):
        record["national_rank"] = position

    selected = {state.upper() for state in states} if states else None
    keep = lambda record: selected is None or record["state"] in selected
    return {
        "rank_by": rank_by,
        "window_years": window_years,
        "base_year": base_year,
        "end_year": END_YEAR,
        "airports_in_national_ranking": len(ranked),
        "ranked": [record for record in ranked if keep(record)],
        "not_ranked": [record for record in not_ranked if keep(record)],
    }


def _apply_scope(routes: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return routes
    if scope == "passenger":
        return routes[routes["AIRCRAFT_CONFIG"] == config.CONFIG_PASSENGER]
    if scope == "scheduled_passenger":
        return routes[
            (routes["AIRCRAFT_CONFIG"] == config.CONFIG_PASSENGER)
            & (routes["CLASS"] == config.CLASS_SCHEDULED)
        ]
    if scope == "freighter":
        return routes[routes["AIRCRAFT_CONFIG"] == config.CONFIG_FREIGHT]
    raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")


def _share(flights: pd.DataFrame, threshold_miles: float) -> float | None:
    total = flights["DEPARTURES_PERFORMED"].sum()
    if total == 0:
        return None
    long_haul = flights.loc[flights["DISTANCE"] >= threshold_miles, "DEPARTURES_PERFORMED"].sum()
    return round(float(100 * long_haul / total), 1)


def long_haul_share(
    routes: pd.DataFrame,
    airport: str,
    threshold_miles: float = config.DEFAULT_LONG_HAUL_MILES,
    airport_summary: pd.DataFrame | None = None,
) -> dict:
    """Share of an airport's departures at or beyond a distance threshold.

    Each route row is weighted by DEPARTURES_PERFORMED, so a route flown daily
    counts far more than one flown once. Passing the all-carrier airport summary
    adds coverage figures and bounds for departures missing from the route file.
    """
    if threshold_miles <= 0:
        raise ValueError("threshold_miles must be positive")
    airport = airport.upper()
    flown = routes[(routes["ORIGIN"] == airport) & (routes["DEPARTURES_PERFORMED"] > 0)]
    if flown.empty:
        raise ValueError(f"No performed departures for {airport} in the route file")

    periods = flown[["YEAR", "MONTH"]].drop_duplicates().sort_values(["YEAR", "MONTH"])
    scopes = {}
    for scope in SCOPES:
        flights = _apply_scope(flown, scope)
        total = float(flights["DEPARTURES_PERFORMED"].sum())
        scopes[scope] = {
            "departures": total,
            "long_haul_departures": float(
                flights.loc[flights["DISTANCE"] >= threshold_miles, "DEPARTURES_PERFORMED"].sum()
            ),
            "share_pct": _share(flights, threshold_miles),
            "sensitivity_pct": {
                int(miles): _share(flights, miles) for miles in config.LONG_HAUL_SENSITIVITY_MILES
            },
        }

    result = {
        "airport": airport,
        "threshold_miles": threshold_miles,
        "period": {
            "first": f"{periods.iloc[0].YEAR}-{periods.iloc[0].MONTH:02d}",
            "last": f"{periods.iloc[-1].YEAR}-{periods.iloc[-1].MONTH:02d}",
            "months": int(len(periods)),
        },
        "weighting": "DEPARTURES_PERFORMED",
        "scopes": scopes,
        "unique_destinations": int(flown["DEST"].nunique()),
        "assumptions": [
            f"Long-haul means DISTANCE of at least {threshold_miles:g} statute miles; "
            "there is no single standard definition.",
            "The route file covers US carriers only.",
            "A partial year cannot be compared with full calendar years, and the "
            "months covered may not represent the whole year.",
        ],
    }
    if airport_summary is not None:
        result["coverage"] = _route_file_coverage(flown, periods, airport, airport_summary, scopes["all"])
    return result


def _route_file_coverage(
    flown: pd.DataFrame,
    periods: pd.DataFrame,
    airport: str,
    airport_summary: pd.DataFrame,
    all_scope: dict,
) -> dict:
    """Compare the route file with the summary for the same airport and months."""
    wanted = set(zip(periods["YEAR"].astype(int), periods["MONTH"].astype(int)))
    rows = airport_summary[airport_summary["airport"] == airport]
    rows = rows[[(y, m) in wanted for y, m in zip(rows["year"].astype(int), rows["month"].astype(int))]]
    if len(rows) != len(wanted):
        return {"available": False, "reason": "Summary does not cover all route-file months"}

    summary_departures = float(rows["departures"].sum())
    summary_passengers = float(rows["passengers"].sum())
    file_departures = all_scope["departures"]
    file_passengers = float(flown["PASSENGERS"].sum())
    missing = max(summary_departures - file_departures, 0.0)

    # The carrier and distance of missing departures are unknown, so the all-flight
    # share is bounded by assuming none or all of them are long-haul.
    long_haul = all_scope["long_haul_departures"]
    denominator = file_departures + missing
    return {
        "available": True,
        "summary_departures": summary_departures,
        "summary_passengers": summary_passengers,
        "departure_coverage_pct": round(100 * file_departures / summary_departures, 1),
        "passenger_coverage_pct": round(100 * file_passengers / summary_passengers, 1),
        "departures_missing_from_route_file": missing,
        "all_flights_share_bounds_pct": {
            "low": round(100 * long_haul / denominator, 1),
            "high": round(100 * (long_haul + missing) / denominator, 1),
        },
    }
