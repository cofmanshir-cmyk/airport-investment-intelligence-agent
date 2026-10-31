"""Regression checks on the real processed snapshot; skipped if it is absent."""

import pandas as pd
import pytest

from src import config, pipeline
from src.analytics import long_haul_share


def load(name):
    path = config.PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not built; run python -m src.pipeline")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def summary():
    return load("t100_airport_month")


@pytest.fixture(scope="module")
def routes():
    return load("t100_routes")


def test_airport_totals_exclude_inbound_on_every_row(summary):
    assert pipeline.airport_row_counting_errors(summary) == 0


def test_every_year_reconciles_to_the_national_series():
    frame = load("counting_reconciliation")
    assert (frame["airport_minus_dom_plus_outbound"] == 0).all()
    assert (frame["national_minus_all_components"] == 0).all()


def test_required_years_have_twelve_months_and_2026_is_partial(summary):
    months = summary.groupby("year")["reporting_month"].nunique()
    for year in config.HISTORY_YEARS:
        assert months[year] == 12
    assert months[2026] == 7


def test_route_file_is_seven_months_of_2026(routes):
    periods = routes[["YEAR", "MONTH"]].drop_duplicates()
    assert set(periods["YEAR"]) == {2026}
    assert sorted(periods["MONTH"]) == list(range(1, 8))


def test_route_file_has_only_us_carrier_data_sources(routes):
    assert set(routes["DATA_SOURCE"]) == {"DU", "IU"}


def test_anc_long_haul_snapshot(routes, summary):
    result = long_haul_share(routes, "ANC", airport_summary=summary)
    assert result["period"]["months"] == 7
    assert result["scopes"]["all"]["departures"] == 43066
    assert result["scopes"]["all"]["share_pct"] == 23.0
    assert result["scopes"]["passenger"]["share_pct"] == 2.7
    assert result["scopes"]["scheduled_passenger"]["share_pct"] == 2.6
    assert result["scopes"]["freighter"]["share_pct"] == 47.7
    assert result["coverage"]["departure_coverage_pct"] == 86.3
    assert result["coverage"]["passenger_coverage_pct"] == 99.4
    assert result["coverage"]["all_flights_share_bounds_pct"] == {"low": 19.8, "high": 33.6}


def test_t100_passengers_track_faa_enplanements_at_hubs(summary):
    airports = load("faa_airports")
    hubs = airports[airports["hub"].isin(["L", "M", "S"])]
    passengers = summary[summary["year"] == 2025].groupby("airport")["passengers"].sum()
    joined = hubs.join(passengers, on="airport", how="inner")
    ratio = joined["passengers"] / joined["enplanements_2025"]
    assert ratio.between(0.99, 1.04).all()
