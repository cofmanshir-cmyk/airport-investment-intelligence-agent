"""Tool checks on the real snapshot; skipped if it is absent."""

import json

import pytest

from src import config, scoring, tools

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "ontime_airport_month.parquet").exists(),
    reason="Processed data not built",
)


def test_raw_metrics_match_hand_calculated_sfo_example():
    metrics = tools.get_airport_profile("SFO")["profile"]["demand_pressure_proxies"]["metrics"]
    assert metrics == {
        "passenger_cagr_3y_pct": 8.87,
        "passenger_cagr_10y_pct": 0.84,
        "absolute_passenger_growth_3y": 5959126,
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


def test_default_ranking_is_still_by_passenger_cagr():
    default = tools.rank_airports(3, [], 25, 100_000)
    explicit = tools.rank_airports(3, [], 25, 100_000, "passenger_cagr")
    assert [row["airport"] for row in default["ranked"]] == [row["airport"] for row in explicit["ranked"]]
    assert default["ranked"][0]["airport"] == "ILG"


def test_ranking_by_absolute_growth_sorts_largest_first_and_keeps_cagr_visible():
    result = tools.rank_airports(3, [], 25, 100_000, "absolute_passenger_growth")
    values = [row["absolute_passenger_growth"] for row in result["ranked"]]
    assert values == sorted(values, reverse=True)
    assert result["ranked"][0]["airport"] == "ORD"
    assert result["ranked"][0]["absolute_passenger_growth"] == 7602196
    assert all(row["passenger_cagr_pct"] is not None for row in result["ranked"])
    assert "Absolute Passenger Growth 3Y" in result["metric"]
    assert "favor larger airports" in " ".join(result["caveats"])


def test_absolute_growth_10y_is_available_and_labelled_as_not_scored():
    result = tools.rank_airports(10, [], 3, 100_000, "absolute_passenger_growth")
    assert "not part of the score" in result["metric"]
    assert result["ranked"][0]["airport"] == "DEN"
    assert result["ranked"][0]["absolute_passenger_growth"] == 13483985


def test_absolute_ranking_keeps_national_rank_under_a_state_filter():
    national = tools.rank_airports(3, [], 25, 100_000, "absolute_passenger_growth")["ranked"]
    regional = tools.rank_airports(3, ["MA"], 25, 100_000, "absolute_passenger_growth")["ranked"]
    bos = next(row for row in regional if row["airport"] == "BOS")
    assert bos["national_rank"] == next(row["national_rank"] for row in national if row["airport"] == "BOS")


def test_profile_shows_score_and_passenger_context():
    section = tools.get_airport_profile("ATL")["profile"]["opportunity_score"]
    assert section["opportunity_score"] == 50.66
    assert section["national_rank"] == 82
    assert section["absolute_passenger_growth_3y"] == 5924745
    assert section["passenger_cagr_3y_pct"] == 4.16
    assert section["passengers_2025"] - section["passengers_2022"] == 5924745
    assert set(section["components"]) == set(scoring.WEIGHTS)


def test_profile_keeps_absolute_growth_10y_outside_the_score():
    proxies = tools.get_airport_profile("DEN")["profile"]["demand_pressure_proxies"]
    assert proxies["supplementary_not_scored"]["absolute_passenger_growth_10y"] == 13483985
    assert "absolute_passenger_growth_10y" not in scoring.WEIGHTS


def test_profile_without_a_score_gives_the_reason_and_keeps_passenger_values():
    section = tools.get_airport_profile("JAC")["profile"]["opportunity_score"]
    assert section["opportunity_score"] is None
    assert section["reason"] == "2022 is incomplete: 11 of 12 months reported"
    assert section["passengers_2025"] == 571901


def test_airport_below_the_threshold_says_so():
    airports = tools._airports()
    small = airports[airports["enplanements_2025"] < 100_000]
    known = set(tools._annual().index.get_level_values("airport"))
    code = sorted(set(small["t100_code"]) & known)[0]
    section = tools.get_airport_profile(code)["profile"]["opportunity_score"]
    assert section["opportunity_score"] is None
    assert "100,000" in section["reason"]


def test_profile_shows_small_base_warnings_first():
    warnings = tools.get_airport_profile("ILG")["profile"]["warnings"]
    assert warnings[0].startswith("Small base year: 2015")
    assert warnings[1].startswith("Small base year: 2022")
    assert any("2015 had 92" in warning for warning in tools.get_airport_profile("PAE")["profile"]["warnings"])


def test_compare_includes_score_and_absolute_growth_differences():
    result = tools.compare_airports(["ATL", "SFO"])
    assert result["differences_from_first_airport"]["opportunity_score"]["SFO_minus_ATL"] == round(62.39 - 50.66, 2)
    demand = result["differences_from_first_airport"]["demand_pressure_proxies"]["SFO_minus_ATL"]
    assert demand["absolute_passenger_growth_3y"] == 5959126 - 5924745


def test_compare_leaves_out_score_difference_when_an_airport_has_no_score():
    result = tools.compare_airports(["ATL", "JAC"])
    assert result["differences_from_first_airport"]["opportunity_score"] == {}


def test_score_ranking_is_national_and_shows_passenger_context():
    result = tools.rank_opportunity_score([], 3)
    assert result["airports_scored_nationally"] == 229
    assert [row["national_rank"] for row in result["ranked"]] == [1, 2, 3]
    first = result["ranked"][0]
    assert first["airport"] == "ILG"
    assert first["opportunity_score"] == 78.38
    for key in ("passengers_2022", "passengers_2025", "absolute_passenger_growth_3y", "passenger_cagr_3y_pct"):
        assert first[key] is not None
    assert first["small_base_warnings"]


def test_geographic_filter_never_changes_a_score_or_rank():
    national = {row["airport"]: row for row in tools.rank_opportunity_score([], 25)["ranked"]}
    regional = tools.rank_opportunity_score(["DE"], 25)["ranked"]
    assert [row["airport"] for row in regional] == ["ILG"]
    assert regional[0]["opportunity_score"] == national["ILG"]["opportunity_score"]
    assert regional[0]["national_rank"] == national["ILG"]["national_rank"]
    assert tools.rank_opportunity_score(["DE"], 25)["normalization_limits"] == tools._scores()["bounds"]


def test_state_filter_also_limits_unscored_airports():
    wyoming = tools.rank_opportunity_score(["WY"], 25)
    assert [row["airport"] for row in wyoming["not_scored"]] == ["JAC"]
    assert tools.rank_opportunity_score(["DE"], 25)["not_scored"] == []


def test_score_results_disclose_the_required_limits():
    for result in (tools.rank_opportunity_score([], 1), tools.get_airport_profile("BOS"), tools.compare_airports(["LAX", "SNA"])):
        text = " ".join([result["disclaimer"], *result["caveats"]])
        assert "35%" in text and "cargo" in text
        assert "favor larger airports" in text
        assert "2022" in text and "recovery" in text
        assert "investment returns" in result["disclaimer"]


def test_load_factor_is_never_called_a_passenger_load_factor():
    text = json.dumps([tools.TOOL_DEFINITIONS, tools.METRIC_DEFINITIONS, tools.PROFILE_CAVEATS]).lower()
    assert "passenger load factor" not in text


def test_route_file_code_maps_to_the_summary_code():
    assert "DJT" in tools._mix().index
    assert "PBI" not in tools._mix().index
    assert tools.get_long_haul_share("DJT", 3000)["airport"] == "DJT"


def test_every_result_carries_sources_and_caveats():
    for result in (
        tools.get_airport_profile("BOS"),
        tools.compare_airports(["LAX", "SNA"]),
        tools.rank_airports(10, [], 3, 250_000),
        tools.rank_opportunity_score([], 3),
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
        ("rank_airports", {"window_years": 3, "states": [], "top_n": 5, "min_enplanements": 100000, "rank_by": "size"}),
        ("rank_opportunity_score", {"states": [], "top_n": 500}),
        ("rank_opportunity_score", {"states": ["XX"], "top_n": 5}),
        ("get_long_haul_share", {"airport": "ANC", "threshold_miles": 0}),
        ("no_such_tool", {}),
    ],
)
def test_bad_input_becomes_a_tool_error(name, tool_input):
    result, is_error = tools.execute_tool(name, tool_input)
    assert is_error and result["error"]


def test_expected_tools_are_exposed():
    names = {tool["name"] for tool in tools.TOOL_DEFINITIONS}
    assert names == {
        "rank_opportunity_score", "rank_airports", "compare_airports", "get_airport_profile", "get_long_haul_share",
    }


def test_rank_by_schema_offers_both_growth_measures():
    schema = next(tool for tool in tools.TOOL_DEFINITIONS if tool["name"] == "rank_airports")["input_schema"]
    assert schema["properties"]["rank_by"]["enum"] == ["absolute_passenger_growth", "passenger_cagr"]


def test_tool_schemas_are_strict():
    for tool in tools.TOOL_DEFINITIONS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
