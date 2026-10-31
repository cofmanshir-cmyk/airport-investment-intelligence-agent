import zipfile

import pandas as pd
import pytest

from src import pipeline


def summary_rows(airport="AAA", year=2025, months=12, **overrides):
    base = {
        "origin_airport_code": airport,
        "origin_airport_name": f"{airport} Airport",
        "total_departures": "10",
        "total_passengers": "1000",
        "total_seats": "1200",
        "total_freight_lbs": "5",
        "domestic_departures": "8",
        "domestic_passengers": "800",
        "domestic_seats": "1000",
        "outbound_international": "2",
        "outbound_international_1": "200",
        "outbound_international_seats": "200",
        "inbound_international": "2",
        "inbound_international_1": "190",
    }
    base.update(overrides)
    return [{**base, "reporting_month": f"{year}-{m:02d}-01T00:00:00.000"} for m in range(1, months + 1)]


def test_blank_international_fields_become_zero_but_totals_do_not():
    rows = summary_rows(months=1)
    del rows[0]["outbound_international"]
    del rows[0]["total_seats"]
    frame = pipeline.normalize_airport_summary(rows)
    assert frame.loc[0, "outbound_intl_departures"] == 0
    assert pd.isna(frame.loc[0, "seats"])


def test_outbound_passenger_field_maps_to_passengers_not_departures():
    frame = pipeline.normalize_airport_summary(summary_rows(months=1))
    assert frame.loc[0, "outbound_intl_departures"] == 2
    assert frame.loc[0, "outbound_intl_passengers"] == 200


def test_duplicate_airport_month_is_rejected():
    rows = summary_rows(months=1) * 2
    with pytest.raises(pipeline.DataValidationError, match="duplicate"):
        pipeline.normalize_airport_summary(rows)


def test_negative_values_are_rejected():
    with pytest.raises(pipeline.DataValidationError, match="Negative"):
        pipeline.normalize_airport_summary(summary_rows(months=1, total_passengers="-1"))


def test_inbound_flights_are_not_counted_in_airport_totals():
    frame = pipeline.normalize_airport_summary(summary_rows(months=2))
    assert pipeline.airport_row_counting_errors(frame) == 0
    # A total that included inbound flights would be flagged on every row.
    inflated = frame.assign(departures=frame["departures"] + frame["inbound_intl_departures"])
    assert pipeline.airport_row_counting_errors(inflated) == 2


def test_reconciliation_shows_national_total_includes_both_directions():
    airports = pipeline.normalize_airport_summary(summary_rows(months=1))
    national = pd.DataFrame({
        "reporting_month": pd.to_datetime(["2025-01-01"]),
        "departures": [12],
        "domestic_departures": [8],
        "outbound_intl_departures": [2],
        "inbound_intl_departures": [2],
    })
    row = pipeline.reconcile_counting(airports, national).iloc[0]
    assert row["airport_minus_dom_plus_outbound"] == 0
    assert row["national_minus_all_components"] == 0
    assert row["national_departures"] - row["airport_departures"] == 2


def history_frame(*spec):
    rows = []
    for airport, year, months, overrides in spec:
        rows += summary_rows(airport, year, months, **overrides)
    return pipeline.normalize_airport_summary(rows)


def test_history_complete_requires_all_base_and_end_years():
    frame = history_frame(("AAA", 2015, 12, {}), ("AAA", 2022, 12, {}), ("AAA", 2025, 12, {}))
    assert bool(pipeline.history_coverage(frame).iloc[0]["history_complete"])


def test_partial_year_is_not_complete():
    frame = history_frame(("AAA", 2015, 11, {}), ("AAA", 2022, 12, {}), ("AAA", 2025, 12, {}))
    row = pipeline.history_coverage(frame).iloc[0]
    assert not bool(row["complete_2015"])
    assert not bool(row["history_complete"])


def test_missing_base_year_is_not_complete():
    frame = history_frame(("AAA", 2022, 12, {}), ("AAA", 2025, 12, {}))
    assert not bool(pipeline.history_coverage(frame).iloc[0]["history_complete"])


def test_zero_passenger_base_year_is_not_complete():
    frame = history_frame(
        ("AAA", 2015, 12, {"total_passengers": "0"}), ("AAA", 2022, 12, {}), ("AAA", 2025, 12, {})
    )
    assert not bool(pipeline.history_coverage(frame).iloc[0]["history_complete"])


def test_partial_2026_never_counts_toward_history():
    frame = history_frame(
        ("AAA", 2015, 12, {}), ("AAA", 2022, 12, {}), ("AAA", 2025, 12, {}), ("AAA", 2026, 7, {})
    )
    coverage = pipeline.history_coverage(frame)
    assert "complete_2026" not in coverage.columns
    assert bool(coverage.iloc[0]["history_complete"])


def route_csv(tmp_path, rows):
    columns = dict.fromkeys(pipeline.ROUTE_COLUMNS, 1)
    columns.update(ORIGIN="ANC", ORIGIN_COUNTRY="US", DEST="SEA", DEST_COUNTRY="US",
                   UNIQUE_CARRIER="AS", UNIQUE_CARRIER_NAME="Alaska", CLASS="F", DATA_SOURCE="DU",
                   YEAR=2026, DISTANCE=1448, DEPARTURES_PERFORMED=5)
    frame = pd.DataFrame([{**columns, **row} for row in rows])
    path = tmp_path / "routes.csv"
    frame.to_csv(path, index=False)
    return path


def test_route_file_profile_reports_actual_coverage(tmp_path):
    path = route_csv(tmp_path, [{"MONTH": 1}, {"MONTH": 7, "DEPARTURES_PERFORMED": 0}])
    routes, profile = pipeline.load_route_file(path)
    assert len(routes) == 2
    assert profile["first_period"] == "2026-01"
    assert profile["last_period"] == "2026-07"
    assert profile["months"] == 2
    assert profile["rows_with_zero_departures"] == 1


def test_route_file_missing_column_is_rejected(tmp_path):
    path = route_csv(tmp_path, [{"MONTH": 1}])
    pd.read_csv(path).drop(columns="DISTANCE").to_csv(path, index=False)
    with pytest.raises(pipeline.DataValidationError, match="DISTANCE"):
        pipeline.load_route_file(path)


def test_faa_parser_drops_non_airport_rows(tmp_path):
    frame = pd.DataFrame({
        "Rank": [1, None], "RO": ["NE", None], "ST": ["MA", None], "Locid": ["BOS", None],
        "City": ["Boston", None], "Airport Name": ["Logan", None], "S/L": ["P", None], "Hub": ["L", None],
        "CY 25 Enplanements": [100, 100], "CY 24 Enplanements": [90, 90], "% Change": [0.1, None],
    })
    path = tmp_path / "faa.xlsx"
    frame.to_excel(path, index=False)
    parsed = pipeline.parse_faa_enplanements(path)
    assert parsed["airport"].tolist() == ["BOS"]
    assert parsed.loc[0, "enplanements_2025"] == 100


def test_ontime_departures_group_by_origin_and_delay_causes_by_destination(tmp_path):
    flights = pd.DataFrame({
        "Year": [2025, 2025], "Month": [1, 1], "Origin": ["LAX", "LAX"], "Dest": ["SNA", "SFO"],
        "Cancelled": [0, 1], "DepDel15": [1, None], "TaxiOut": [20, None], "ArrDel15": [1, None],
        "CarrierDelay": [0, None], "WeatherDelay": [0, None], "NASDelay": [30, None],
        "SecurityDelay": [0, None], "LateAircraftDelay": [0, None],
    })
    path = tmp_path / "ontime.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flights.csv", flights.to_csv(index=False))
    result = pipeline.aggregate_ontime_month(path).set_index("airport")
    assert result.loc["LAX", "departures"] == 2
    assert result.loc["LAX", "cancelled"] == 1
    assert result.loc["LAX", "dep_del15_reported"] == 1
    assert result.loc["LAX", "taxi_out_min"] == 20
    assert result.loc["LAX", "nas_delay_min"] == 0
    assert result.loc["SNA", "nas_delay_min"] == 30
    assert result.loc["SNA", "departures"] == 0
