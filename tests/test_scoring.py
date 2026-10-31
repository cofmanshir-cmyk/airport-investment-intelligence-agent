"""Airport Opportunity Score checks. Made-up airports need no data; snapshot checks use local files."""

import random
import re
from pathlib import Path

import pandas as pd
import pytest

from src import analytics, config, scoring, tools

has_snapshot = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "t100_airport_month.parquet").exists(),
    reason="Processed data not built",
)


def year_rows(airport, year, passengers, departures, seats, months=12, **month_overrides):
    """Build monthly rows that add up to the given annual totals."""
    rows = []
    for month in range(1, months + 1):
        row = {
            "airport": airport,
            "year": year,
            "reporting_month": pd.Timestamp(year, month, 1),
            "passengers": passengers / months,
            "departures": departures / months,
            "seats": seats / months,
            "freight_lbs": 0.0,
            "domestic_departures": departures / months,
        }
        row.update(month_overrides.get(f"m{month}", {}))
        rows.append(row)
    return rows


def airport_rows(airport, index, **overrides):
    """Give each made-up airport different growth, using its position only."""
    base = 120_000 + 12_000 * index
    rows = (
        year_rows(airport, 2015, base, 1_200 + 12 * index, 150_000 + 15_000 * index)
        + year_rows(airport, 2022, base * (1.1 + 0.02 * index), 1_300 + 13 * index, 160_000 + 16_000 * index)
        + year_rows(airport, 2025, base * (1.3 + 0.05 * index), 1_400 + 10 * index, 170_000 + 17_000 * index)
    )
    return rows


def national(codes, extra_rows=()):
    rows = [row for index, code in enumerate(codes) for row in airport_rows(code, index)]
    annual = analytics.annual_totals(pd.DataFrame([*rows, *extra_rows]))
    airports = pd.DataFrame({
        "t100_code": list(codes),
        "airport_name": [f"{code} Airport" for code in codes],
        "state": ["AA"] * len(codes),
        "hub": ["S"] * len(codes),
        "enplanements_2025": [200_000] * len(codes),
    })
    return annual, airports


CODES = [f"Q{chr(65 + a)}{chr(65 + b)}" for a in range(4) for b in range(5)]


def test_there_are_11_weights_totalling_100():
    assert len(scoring.WEIGHTS) == 11
    assert sum(scoring.WEIGHTS.values()) == 100


def test_weights_match_the_approved_methodology():
    assert scoring.WEIGHTS == {
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


def test_cargo_inclusive_metrics_carry_35_points():
    assert scoring.CARGO_INCLUSIVE_POINTS == 35


@pytest.mark.parametrize(
    ("value", "expected"), [(-50, 0), (10, 0), (15, 25), (20, 50), (30, 100), (500, 100)]
)
def test_normalize_clips_then_scales_linearly(value, expected):
    assert scoring.normalize(value, 10, 30) == expected


def test_bounds_use_linear_interpolation_at_the_5th_and_95th_percentiles():
    # Three values: positions 0.1 and 1.9, so 10 + 0.1 x 10 and 20 + 0.9 x 20.
    assert scoring.normalization_bounds(pd.Series([10.0, 20.0, 40.0])) == pytest.approx((11.0, 38.0))


def test_equal_limits_stop_with_an_error_instead_of_guessing():
    with pytest.raises(ValueError, match="limits are equal"):
        scoring.normalization_bounds(pd.Series([5.0, 5.0, 5.0], name="metric"))


def test_metrics_match_a_hand_calculation():
    rows = (
        year_rows("QQQ", 2015, 1_000_000, 10_000, 1_250_000)
        + year_rows("QQQ", 2022, 1_000_000, 8_000, 1_250_000)
        + year_rows("QQQ", 2025, 1_331_000, 10_648, 1_597_200)
    )
    metrics = analytics.metric_values(analytics.annual_totals(pd.DataFrame(rows)), "QQQ")
    assert metrics["absolute_passenger_growth_3y"] == 331_000
    assert metrics["passenger_cagr_3y_pct"] == pytest.approx(10.0)
    assert metrics["passenger_cagr_10y_pct"] == pytest.approx((1.331 ** 0.1 - 1) * 100)
    assert metrics["flight_cagr_3y_pct"] == pytest.approx(10.0)
    assert metrics["growth_gap_3y_pts"] == pytest.approx(0.0)
    assert metrics["passengers_per_flight_growth_3y_pct"] == pytest.approx(0.0)
    assert metrics["total_load_factor_2025_pct"] == pytest.approx(100 * 1_331_000 / 1_597_200)
    assert metrics["load_factor_change_3y_pts"] == pytest.approx(100 * 1_331_000 / 1_597_200 - 80)


def test_score_applies_the_formula_to_unrounded_values():
    annual, airports = national(CODES)
    result = scoring.score_airports(annual, airports)
    peers = pd.DataFrame({code: analytics.metric_values(annual, code) for code in CODES}).T.astype(float)
    for record in result["scored"]:
        expected = 0.0
        for metric, weight in scoring.WEIGHTS.items():
            lower, upper = peers[metric].quantile(0.05), peers[metric].quantile(0.95)
            clipped = min(max(peers.loc[record["airport"], metric], lower), upper)
            expected += weight * 100 * (clipped - lower) / (upper - lower) / 100
        assert record["opportunity_score"] == pytest.approx(expected, abs=0.005)


def test_records_show_passenger_context_beside_the_score():
    annual, airports = national(CODES)
    record = scoring.score_airports(annual, airports)["scored"][0]
    inputs = analytics.year_inputs(annual, record["airport"])
    assert record["passengers_2022"] == inputs["2022"]["passengers"]
    assert record["passengers_2025"] == inputs["2025"]["passengers"]
    assert record["absolute_passenger_growth_3y"] == record["passengers_2025"] - record["passengers_2022"]
    assert record["passenger_cagr_3y_pct"] == analytics.raw_metrics(annual, record["airport"])["passenger_cagr_3y_pct"]
    assert set(record["components"]) == set(scoring.WEIGHTS)


def test_missing_month_gives_no_score_and_a_reason():
    rows = (
        year_rows("QZZ", 2015, 500_000, 5_000, 600_000)
        + year_rows("QZZ", 2022, 600_000, 6_000, 700_000, months=11)
        + year_rows("QZZ", 2025, 700_000, 7_000, 800_000)
    )
    annual, airports = national(CODES, rows)
    airports = pd.concat([airports, airports.iloc[[0]].assign(t100_code="QZZ")], ignore_index=True)
    result = scoring.score_airports(annual, airports)
    assert "QZZ" not in {record["airport"] for record in result["scored"]}
    assert [(row["airport"], row["reason"]) for row in result["not_scored"]] == [
        ("QZZ", "2022 is incomplete: 11 of 12 months reported")
    ]
    assert result["airports_eligible"] == len(CODES) + 1
    assert result["airports_scored"] == len(CODES)


def test_excluded_airport_does_not_affect_peer_limits():
    annual, airports = national(CODES)
    without = scoring.score_airports(annual, airports)
    rows = year_rows("QZZ", 2025, 90_000_000, 7_000, 100_000_000)
    annual_more, airports_more = national(CODES, rows)
    airports_more = pd.concat([airports_more, airports_more.iloc[[0]].assign(t100_code="QZZ")], ignore_index=True)
    with_excluded = scoring.score_airports(annual_more, airports_more)
    assert with_excluded["bounds"] == without["bounds"]


def test_blank_month_is_missing_not_zero():
    rows = year_rows("QQQ", 2022, 600_000, 6_000, 700_000, m3={"seats": None})
    annual = analytics.annual_totals(pd.DataFrame(rows))
    assert analytics.complete_year(annual, "QQQ", 2022) is None


def test_reported_zero_passenger_month_is_a_genuine_zero():
    rows = year_rows("QQQ", 2022, 600_000, 6_000, 700_000, m3={"passengers": 0.0})
    annual = analytics.annual_totals(pd.DataFrame(rows))
    assert analytics.complete_year(annual, "QQQ", 2022) is not None


def test_tied_scores_break_by_airport_code():
    rows = [
        row
        for code in ("QXB", "QXA", "QXC")
        for row in (
            year_rows(code, 2015, 100_000, 1_000, 120_000)
            + year_rows(code, 2022, 110_000, 1_100, 130_000)
            + year_rows(code, 2025, 120_000, 1_200, 140_000)
        )
    ]
    annual, airports = national(CODES, rows)
    airports = pd.concat(
        [airports, *[airports.iloc[[0]].assign(t100_code=code) for code in ("QXB", "QXA", "QXC")]], ignore_index=True
    )
    scored = scoring.score_airports(annual, airports)["scored"]
    tied = [record for record in scored if record["airport"] in ("QXA", "QXB", "QXC")]
    assert [record["airport"] for record in tied] == ["QXA", "QXB", "QXC"]
    assert len({record["opportunity_score"] for record in tied}) == 1
    assert [record["national_rank"] for record in tied] == list(range(tied[0]["national_rank"], tied[0]["national_rank"] + 3))


def test_results_do_not_depend_on_input_order():
    annual, airports = national(CODES)
    shuffled = airports.sample(frac=1, random_state=random.Random(7).randint(0, 1000))
    assert scoring.score_airports(annual, shuffled) == scoring.score_airports(annual, airports)


def test_scores_follow_the_data_when_airports_are_renamed():
    annual, airports = national(CODES)
    renamed_codes = [f"Z{code[1:]}" for code in CODES]
    annual_renamed, airports_renamed = national(renamed_codes)
    first = scoring.score_airports(annual, airports)["scored"]
    second = scoring.score_airports(annual_renamed, airports_renamed)["scored"]
    assert [(row["airport"][1:], row["opportunity_score"]) for row in first] == [
        (row["airport"][1:], row["opportunity_score"]) for row in second
    ]


@has_snapshot
def test_scoring_code_names_no_airport():
    source = Path(scoring.__file__).read_text()
    codes = set(tools._airports()["t100_code"]) | set(tools._airports()["airport"])
    tokens = set(re.findall(r"\b[A-Z0-9]{3}\b", source))
    assert not tokens & codes


@has_snapshot
def test_snapshot_reproduces_the_approved_comparison():
    scores = tools._scores()
    by_code = {record["airport"]: record for record in scores["scored"]}
    assert scores["airports_eligible"] == 233
    assert scores["airports_scored"] == 229
    assert scores["scored"][0]["national_rank"] == 1
    assert by_code["ILG"]["opportunity_score"] == 78.38
    assert by_code["ATL"]["opportunity_score"] == 50.66
    assert by_code["ATL"]["national_rank"] == 82
    assert by_code["ACY"]["opportunity_score"] == 8.85
    assert by_code["ACY"]["national_rank"] == 229
    assert by_code["ATL"]["components"]["absolute_passenger_growth_3y"]["points"] == 10.0
    limits = scores["bounds"]["absolute_passenger_growth_3y"]
    assert limits["lower"] == pytest.approx(-14073.2)
    assert limits["upper"] == pytest.approx(3232419.4)


@has_snapshot
def test_snapshot_reports_unscored_airports_with_reasons():
    assert {row["airport"]: row["reason"] for row in tools._scores()["not_scored"]} == {
        "HDN": "2015 is incomplete: 11 of 12 months reported",
        "JAC": "2022 is incomplete: 11 of 12 months reported",
        "VRB": "2015 is incomplete: 4 of 12 months reported",
        "XWA": "2015 is incomplete: 0 of 12 months reported",
    }


@has_snapshot
def test_small_base_airports_are_flagged_and_still_scored():
    by_code = {record["airport"]: record for record in tools._scores()["scored"]}
    assert any("2022 had 4,988 passengers" in warning for warning in by_code["ILG"]["small_base_warnings"])
    assert any("2015 had 92 passengers" in warning for warning in by_code["PAE"]["small_base_warnings"])
    assert by_code["ILG"]["opportunity_score"] is not None
    assert by_code["PAE"]["opportunity_score"] is not None


@has_snapshot
def test_every_small_base_year_in_the_data_is_flagged():
    annual = tools._annual()
    for record in tools._scores()["scored"]:
        expected = [
            year
            for year in (config.BASE_YEAR_10Y, config.BASE_YEAR_3Y)
            if annual.loc[(record["airport"], year), "passengers"] < config.SMALL_BASE_WARNING_PASSENGERS
        ]
        assert len(record["small_base_warnings"]) == len(expected)
