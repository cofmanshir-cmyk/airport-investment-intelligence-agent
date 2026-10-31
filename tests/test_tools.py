"""Tool checks on the real snapshot; skipped if it is absent."""

import json

import pytest

from src import analytics, config, tools

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "ontime_airport_month.parquet").exists(),
    reason="Processed data not built",
)


def test_raw_metrics_match_hand_calculated_sfo_example():
    metrics = tools.get_airport_profile("SFO")["profile"]["demand_pressure_proxies"]["metrics"]
    assert metrics == {
        "passenger_cagr_3y_pct": 8.87,
        "passenger_cagr_10y_pct": 0.84,
        "flight_cagr_3y_pct": 5.96,
        "flight_cagr_10y_pct": -0.41,
        "growth_gap_3y_pts": 2.91,
        "total_load_factor_2025_pct": 82.26,
        "load_factor_change_3y_pts": 0.19,
        "passengers_per_flight_growth_3y_pct": 2.74,
        "flight_growth_acceleration_pts": 6.37,
        "passenger_growth_acceleration_pts": 8.03,
    }


def test_incomplete_base_year_gives_null_not_an_estimate():
    profile = tools.get_airport_profile("JAC")["profile"]["demand_pressure_proxies"]
    assert profile["metrics"]["passenger_cagr_3y_pct"] is None
    assert profile["metrics"]["passenger_cagr_10y_pct"] is not None
    assert profile["annual_inputs"]["2022"]["months_reported"] == 11


def test_no_airport_is_lost_by_code_mismatch_at_the_demo_threshold():
    airports = tools._airports()
    known = set(tools._annual().index.get_level_values("airport"))
    eligible = airports[airports["enplanements_2025"] >= config.DEFAULT_MIN_ENPLANEMENTS]
    assert eligible[~eligible["t100_code"].isin(known)].empty


def test_ranking_counts_match_the_reviewed_universe():
    result = tools.rank_airports(3, [], 5, 100_000)
    assert result["airports_in_national_ranking"] + len(result["not_ranked"]) == 233
    assert len(result["ranked"]) == 5


def test_ranking_is_sorted_and_flags_small_base():
    ranked = tools.rank_airports(3, [], 25, 100_000)["ranked"]
    values = [row["passenger_cagr_pct"] for row in ranked]
    assert values == sorted(values, reverse=True)
    assert ranked[0]["airport"] == "ILG"
    assert any("Small base" in warning for warning in ranked[0]["warnings"])


def test_new_england_filter_keeps_national_rank():
    result = tools.rank_airports(3, ["CT", "MA", "ME", "NH", "RI", "VT"], 25, 100_000)
    assert {row["airport"] for row in result["ranked"]} == {
        "BOS", "BDL", "PVD", "PWM", "HVN", "BTV", "MHT", "BGR", "ACK", "ORH"
    }
    assert result["airports_in_national_ranking"] > len(result["ranked"])
    ranks = [row["national_rank"] for row in result["ranked"]]
    assert ranks == sorted(ranks)


def test_incomplete_airport_is_reported_as_not_ranked():
    result = tools.rank_airports(3, ["WY"], 25, 100_000)
    assert "JAC" in {row["airport"] for row in result["not_ranked"]}


def test_compare_keeps_congestion_apart_and_differences_are_exact():
    result = tools.compare_airports(["LAX", "SNA"])
    lax = result["airports"]["LAX"]["operational_congestion"]
    sna = result["airports"]["SNA"]["operational_congestion"]
    difference = result["differences_from_first_airport"]["operational_congestion"]["SNA_minus_LAX"]
    assert lax["departures_delayed_15_plus_min_pct"] == 18.1
    assert sna["departures_delayed_15_plus_min_pct"] == 19.2
    assert difference["departures_delayed_15_plus_min_pct"] == 1.1
    assert "passenger_cagr_3y_pct" not in difference


def test_every_result_carries_sources_and_caveats():
    for result in (
        tools.get_airport_profile("BOS"),
        tools.compare_airports(["LAX", "SNA"]),
        tools.rank_airports(10, [], 3, 250_000),
        tools.get_long_haul_share("ANC", 3000),
    ):
        assert result["sources"] and result["caveats"] and result["definitions"]
        json.dumps(result, default=float)


def test_long_haul_tool_returns_the_snapshot_values():
    result = tools.get_long_haul_share("anc", 3000)
    assert result["scopes"]["scheduled_passenger"]["share_pct"] == 2.6
    assert result["scopes"]["all"]["share_pct"] == 23.0


def test_faa_code_is_accepted():
    assert tools.get_airport_profile("IWA")["profile"]["airport"] == "AZA"


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("get_airport_profile", {"airport": "ZZZZ"}),
        ("compare_airports", {"airports": ["LAX"]}),
        ("compare_airports", {"airports": ["LAX", "lax"]}),
        ("rank_airports", {"window_years": 5, "states": [], "top_n": 5, "min_enplanements": 100000}),
        ("rank_airports", {"window_years": 3, "states": [], "top_n": 500, "min_enplanements": 100000}),
        ("rank_airports", {"window_years": 3, "states": ["XX"], "top_n": 5, "min_enplanements": 100000}),
        ("rank_airports", {"window_years": 3, "states": [], "top_n": 5, "min_enplanements": 1}),
        ("get_long_haul_share", {"airport": "ANC", "threshold_miles": 0}),
        ("no_such_tool", {}),
    ],
)
def test_bad_input_becomes_a_tool_error(name, tool_input):
    result, is_error = tools.execute_tool(name, tool_input)
    assert is_error and result["error"]


def test_no_weighted_score_is_exposed():
    names = {tool["name"] for tool in tools.TOOL_DEFINITIONS}
    assert names == {"rank_airports", "compare_airports", "get_airport_profile", "get_long_haul_share"}
    assert not hasattr(analytics, "opportunity_score")


def test_tool_schemas_are_strict():
    for tool in tools.TOOL_DEFINITIONS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
