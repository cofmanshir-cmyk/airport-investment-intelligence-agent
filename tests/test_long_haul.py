import pandas as pd
import pytest

from src.analytics import long_haul_share


def route(dest, distance, departures, config=1, cls="F", passengers=0.0, origin="ANC", month=1):
    return {
        "YEAR": 2026, "MONTH": month, "ORIGIN": origin, "DEST": dest, "DISTANCE": distance,
        "DEPARTURES_PERFORMED": departures, "AIRCRAFT_CONFIG": config, "CLASS": cls,
        "PASSENGERS": passengers,
    }


@pytest.fixture
def routes():
    return pd.DataFrame([
        route("SEA", 1448, 90, passengers=9000),   # frequent short passenger route
        route("JFK", 3386, 10, passengers=1000),   # infrequent long passenger route
        route("HKG", 5081, 60, config=2, cls="G"),  # long freighter route
        route("FAI", 261, 40, config=2, cls="G"),   # short freighter route
        route("ORD", 2846, 0),                      # listed but never flown
        route("ZZZ", 9000, 50, origin="SEA"),       # another airport's departures
    ])


def test_weights_by_departures_not_routes(routes):
    passenger = long_haul_share(routes, "ANC")["scopes"]["passenger"]
    # 10 of 100 departures are long-haul; counting routes would give 1 in 2.
    assert passenger["share_pct"] == 10.0
    assert passenger["departures"] == 100


def test_all_flights_scope(routes):
    assert long_haul_share(routes, "ANC")["scopes"]["all"]["share_pct"] == round(100 * 70 / 200, 1)


def test_freighter_scope(routes):
    assert long_haul_share(routes, "ANC")["scopes"]["freighter"]["share_pct"] == 60.0


def test_scheduled_passenger_excludes_charters():
    frame = pd.DataFrame([route("SEA", 1448, 10), route("LAS", 3000, 10, cls="L")])
    scopes = long_haul_share(frame, "ANC")["scopes"]
    assert scopes["passenger"]["share_pct"] == 50.0
    assert scopes["scheduled_passenger"]["share_pct"] == 0.0


def test_threshold_is_inclusive(routes):
    frame = pd.DataFrame([route("A", 3000, 1), route("B", 2999, 1)])
    assert long_haul_share(frame, "ANC", threshold_miles=3000)["scopes"]["all"]["share_pct"] == 50.0


def test_threshold_is_a_parameter(routes):
    assert long_haul_share(routes, "ANC", threshold_miles=1000)["scopes"]["passenger"]["share_pct"] == 100.0


def test_sensitivity_is_reported(routes):
    sensitivity = long_haul_share(routes, "ANC")["scopes"]["passenger"]["sensitivity_pct"]
    assert sensitivity == {1500: 10.0, 2400: 10.0, 3000: 10.0, 4000: 0.0}


def test_unflown_routes_are_excluded(routes):
    assert long_haul_share(routes, "ANC")["unique_destinations"] == 4


def test_empty_scope_returns_none_rather_than_zero():
    frame = pd.DataFrame([route("SEA", 1448, 10)])
    assert long_haul_share(frame, "ANC")["scopes"]["freighter"]["share_pct"] is None


def test_unknown_airport_raises(routes):
    with pytest.raises(ValueError, match="XXX"):
        long_haul_share(routes, "XXX")


def test_invalid_threshold_raises(routes):
    with pytest.raises(ValueError, match="positive"):
        long_haul_share(routes, "ANC", threshold_miles=0)


def test_period_reports_partial_coverage():
    frame = pd.DataFrame([route("A", 100, 1, month=1), route("A", 100, 1, month=7)])
    assert long_haul_share(frame, "ANC")["period"] == {"first": "2026-01", "last": "2026-07", "months": 2}


def summary(departures, passengers, month=1, airport="ANC"):
    return {"airport": airport, "year": 2026, "month": month, "departures": departures, "passengers": passengers}


def test_coverage_and_bounds_for_missing_departures(routes):
    result = long_haul_share(routes, "ANC", airport_summary=pd.DataFrame([summary(250, 11000)]))
    coverage = result["coverage"]
    assert coverage["departure_coverage_pct"] == 80.0
    assert coverage["passenger_coverage_pct"] == round(100 * 10000 / 11000, 1)
    assert coverage["departures_missing_from_route_file"] == 50
    # 70 known long-haul of 200; 50 more departures of unknown distance.
    assert coverage["all_flights_share_bounds_pct"] == {"low": 28.0, "high": 48.0}


def test_coverage_unavailable_when_summary_lacks_a_month():
    frame = pd.DataFrame([route("A", 100, 1, month=1), route("A", 100, 1, month=2)])
    coverage = long_haul_share(frame, "ANC", airport_summary=pd.DataFrame([summary(5, 5)]))["coverage"]
    assert coverage["available"] is False
