"""Checks that behaviour comes from the data and rules, never from a named airport, region or question."""

import ast
import re

import pandas as pd
import pytest

from src import agent, analytics, config, presentation, scoring, tools
from src.agent import ToolCall

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "t100_airport_month.parquet").exists(),
    reason="Processed data not built",
)

PRODUCTION = ["app.py", *[f"src/{name}.py" for name in ("agent", "cli", "gemini_agent", "grounding", "presentation", "scoring", "tools", "analytics")]]
GROWTH = "rank_airports"
NEUTRAL_TEXT = "An answer that states no score."


def source(name):
    return (config.ROOT / name).read_text()


def known_codes():
    airports = tools._airports()
    return (
        set(airports["t100_code"])
        | set(airports["airport"])
        | set(tools._annual().index.get_level_values("airport"))
    )


def relabelled():
    """Give every airport a meaningless code, in an order unrelated to the original."""
    annual, airports = tools._annual(), tools._airports()
    codes = sorted(set(annual.index.get_level_values("airport")) | set(airports["t100_code"]))
    mapping = {code: f"Z{position:05d}" for position, code in enumerate(reversed(codes))}
    return mapping, annual.rename(index=mapping, level="airport").sort_index(), airports.assign(
        t100_code=airports["t100_code"].map(mapping)
    )


def regions():
    """Every single state with scored airports, plus groups of neighbours in alphabetical order."""
    scored_states = sorted({record["state"] for record in tools._scores()["scored"]})
    groups = [[state] for state in scored_states]
    groups += [scored_states[start:start + 6] for start in range(0, len(scored_states), 6)]
    return groups


def test_scores_do_not_depend_on_airport_codes():
    original = scoring.score_airports(tools._annual(), tools._airports())
    mapping, annual, airports = relabelled()
    renamed = scoring.score_airports(annual, airports)
    assert original["airports_scored"] > 0
    assert {mapping[row["airport"]]: row["opportunity_score"] for row in original["scored"]} == {
        row["airport"]: row["opportunity_score"] for row in renamed["scored"]
    }
    assert original["bounds"] == renamed["bounds"]


@pytest.mark.parametrize("window", [3, 10])
def test_growth_rankings_do_not_depend_on_airport_codes(window):
    mapping, annual, airports = relabelled()
    mix = tools._mix().rename(index=mapping)
    threshold = config.DEFAULT_MIN_ENPLANEMENTS
    original = analytics.rank_passenger_growth(tools._annual(), tools._airports(), tools._mix(), window, threshold)
    renamed = analytics.rank_passenger_growth(annual, airports, mix, window, threshold)
    assert original["ranked"]
    assert {mapping[row["airport"]]: row["passenger_cagr_pct"] for row in original["ranked"]} == {
        row["airport"]: row["passenger_cagr_pct"] for row in renamed["ranked"]
    }


def test_filtered_score_lists_follow_the_national_order_in_every_region():
    national = tools._scores()["scored"]
    for states in regions():
        result = tools.rank_opportunity_score(states, 5)
        expected = [row for row in national if row["state"] in states][:5]
        assert [row["airport"] for row in result["ranked"]] == [row["airport"] for row in expected], states
        assert [row["national_rank"] for row in result["ranked"]] == [row["national_rank"] for row in expected], states
        assert [row["opportunity_score"] for row in result["ranked"]] == [row["opportunity_score"] for row in expected], states


def test_table_and_follow_up_rules_hold_in_every_region():
    checked_both = 0
    for states in regions():
        score_result = tools.rank_opportunity_score(states, 5)
        growth_result = tools.rank_airports(3, states, 5, config.DEFAULT_MIN_ENPLANEMENTS)
        score = ToolCall("rank_opportunity_score", {}, score_result, False)
        growth = ToolCall(GROWTH, {}, growth_result, False)
        score_codes = [row["airport"] for row in score_result["ranked"]]
        growth_codes = [row["airport"] for row in growth_result["ranked"]]

        for calls in ([score, growth], [growth, score]):
            assert agent.main_set(calls) == score_codes, states
            assert presentation.main_tables(calls)[0].title == presentation.SCORE_TABLE_TITLE, states
            assert presentation.reports_opportunity_score(calls, NEUTRAL_TEXT), states
            if growth_codes:
                assert agent.main_set(calls, GROWTH) == growth_codes, states
                assert presentation.main_tables(calls, GROWTH)[0].title.startswith("Ranked by Passenger CAGR"), states
                assert len(presentation.main_tables(calls, GROWTH)) == 1, states
                assert not presentation.reports_opportunity_score(calls, NEUTRAL_TEXT, preferred=GROWTH), states
        checked_both += bool(growth_codes)
    assert checked_both > 10


def test_main_table_matches_the_featured_airport_in_every_region():
    checked = 0
    for states in regions():
        ranked = tools.rank_opportunity_score(states, 2)["ranked"]
        if len(ranked) < 2:
            continue
        codes = [row["airport"] for row in ranked]
        calls = [ToolCall("get_airport_profile", {}, tools.get_airport_profile(code, ["score", "demand"]), False) for code in codes]
        for featured in codes:
            tables = presentation.main_tables(calls, None, f"{featured} is where I would start.")
            assert len(tables) == 1, states
            assert featured in tables[0].data.columns, states
            assert not set(codes) - {featured} & set(tables[0].data.columns), states
        checked += 1
    assert checked > 10


def test_named_airport_decides_the_table_in_every_region():
    checked = 0
    for states in regions():
        ranked = tools.rank_opportunity_score(states, 2)["ranked"]
        if len(ranked) < 2:
            continue
        first, second = [row["airport"] for row in ranked]
        calls = [ToolCall("get_airport_profile", {}, tools.get_airport_profile(code, ["score"]), False) for code in (first, second)]
        text = f"{first} is notable, but {second} is the one to review."
        columns = presentation.main_tables(calls, second, text)[0].data.columns
        assert second in columns and first not in columns, states
        assert agent.next_main_set([first, second, "ZZZ"], calls, second) == [first, second, "ZZZ"], states
        checked += 1
    assert checked > 10


def test_requested_section_decides_the_evidence_in_every_region():
    sections = {
        "congestion": "Operational congestion",
        "cargo": "Cargo context, not scored",
        "demand": presentation.DEMAND_TABLE_TITLE,
        "score": presentation.SCORE_TABLE_TITLE,
    }
    checked = 0
    for states in regions():
        ranked = tools.rank_opportunity_score(states, 2)["ranked"]
        if len(ranked) < 2:
            continue
        codes = [row["airport"] for row in ranked]
        for name, title in sections.items():
            call = ToolCall("compare_airports", {}, tools.compare_airports(codes, [name]), False)
            evidence = presentation.build_evidence(call)
            focused = presentation.focused_evidence(evidence, call, [call], "A reply with no figures.")
            shown = [table.title for table in focused.tables]
            if evidence.tables and title in [table.title for table in evidence.tables]:
                assert title in shown, (states, name)
                assert not (set(sections.values()) - {title}) & set(shown), (states, name)
            assert focused.raw is call.result
        checked += 1
    assert checked > 10


def _exact_congestion(code):
    rows = pq_read("ontime_airport_month")
    rows = rows[rows["airport"] == code]
    total = rows.drop(columns=["airport", "year", "month"]).sum()
    return {
        "departures_delayed_15_plus_min_pct": 100 * total["dep_del15"] / total["dep_del15_reported"],
        "cancelled_departures_pct": 100 * total["cancelled"] / total["departures"],
        "average_taxi_out_min": total["taxi_out_min"] / total["taxi_out_reported"],
        "arrivals_delayed_15_plus_min_pct": 100 * total["arr_del15"] / total["arr_del15_reported"],
        "nas_delay_min_per_arrival": total["nas_delay_min"] / total["arrivals"],
        "weather_delay_min_per_arrival": total["weather_delay_min"] / total["arrivals"],
        "carrier_delay_min_per_arrival": total["carrier_delay_min"] / total["arrivals"],
        "late_aircraft_delay_min_per_arrival": total["late_aircraft_delay_min"] / total["arrivals"],
    }


def pq_read(name):
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")


@pytest.mark.parametrize(
    "measure, rounded",
    [
        ("departures_delayed_15_plus_min_pct", 1.1),
        ("cancelled_departures_pct", 0.4),
        ("average_taxi_out_min", -1.7),
        ("arrivals_delayed_15_plus_min_pct", -0.3),
        ("nas_delay_min_per_arrival", -0.2),
        ("weather_delay_min_per_arrival", -0.2),
        ("carrier_delay_min_per_arrival", -1.5),
        ("late_aircraft_delay_min_per_arrival", -1.0),
    ],
)
def test_comparison_differences_come_from_unrounded_values(measure, rounded):
    result = tools.compare_airports(["LAX", "SNA"], ["congestion"])
    shown = result["differences_from_first_airport"]["operational_congestion"]["SNA_minus_LAX"][measure]
    exact = _exact_congestion("SNA")[measure] - _exact_congestion("LAX")[measure]
    assert shown == rounded
    assert shown == round(exact, 1)


def test_rounding_the_differences_leaves_airport_values_unchanged():
    result = tools.compare_airports(["LAX", "SNA"], ["congestion"])
    lax, sna = (result["airports"][code]["operational_congestion"] for code in ("LAX", "SNA"))
    assert (lax["average_taxi_out_min"], sna["average_taxi_out_min"]) == (17.7, 16.1)
    assert (lax["departures_delayed_15_plus_min_pct"], sna["departures_delayed_15_plus_min_pct"]) == (18.1, 19.2)
    assert (lax["cancelled_departures_pct"], sna["cancelled_departures_pct"]) == (0.7, 1.0)
    assert (lax["nas_delay_min_per_arrival"], sna["nas_delay_min_per_arrival"]) == (2.0, 1.7)
    assert (lax["carrier_delay_min_per_arrival"], sna["carrier_delay_min_per_arrival"]) == (5.4, 3.8)
    assert (lax["departures_in_sample"], lax["share_of_t100_domestic_departures_pct"]) == (190472, 91.1)
    assert "calculated before rounding" in result["interpretation_note"]


def test_every_comparison_difference_matches_unrounded_values_across_regions():
    checked = 0
    for states in regions():
        ranked = tools.rank_opportunity_score(states, 2)["ranked"]
        if len(ranked) < 2:
            continue
        first, second = [row["airport"] for row in ranked]
        result = tools.compare_airports([first, second], ["congestion"])
        if not (result["airports"][first]["operational_congestion"] and result["airports"][second]["operational_congestion"]):
            continue
        shown = result["differences_from_first_airport"]["operational_congestion"][f"{second}_minus_{first}"]
        for measure, value in _exact_congestion(second).items():
            base = _exact_congestion(first)[measure]
            if measure in shown:
                assert shown[measure] == round(value - base, 1) + 0, (states, measure)
        demand = result["differences_from_first_airport"]["demand_pressure_proxies"][f"{second}_minus_{first}"]
        values = [analytics.metric_values(tools._annual(), code) for code in (first, second)]
        for measure, difference in demand.items():
            assert difference == round(values[1][measure] - values[0][measure], 2) + 0, (states, measure)
        checked += 1
    assert checked > 5


def test_long_haul_table_matches_the_route_file_in_every_size_of_airport():
    routes = pq_read("t100_routes")
    routes = routes[routes["DEPARTURES_PERFORMED"] > 0]
    busiest = routes.groupby("ORIGIN")["DEPARTURES_PERFORMED"].sum().sort_values(ascending=False)
    freight = routes[routes["AIRCRAFT_CONFIG"] == config.CONFIG_FREIGHT].groupby("ORIGIN")["DEPARTURES_PERFORMED"].sum()
    summary_codes = set(tools._annual().index.get_level_values("airport"))
    candidates = [code for code in busiest.index if code in summary_codes][:40]
    for code in candidates:
        result = tools.get_long_haul_share(code, 3000)
        table = presentation.main_tables([ToolCall("get_long_haul_share", {}, result, False)])[0]
        rows = table.data.set_index("Flights")
        flown = routes[routes["ORIGIN"] == code]
        assert rows.loc["All flights in file", "Departures"] == flown["DEPARTURES_PERFORMED"].sum()
        passenger = flown[flown["AIRCRAFT_CONFIG"] == config.CONFIG_PASSENGER]
        assert rows.loc["Passenger aircraft", "Departures"] == passenger["DEPARTURES_PERFORMED"].sum()
        assert rows.loc["Freighters", "Departures"] == freight.get(code, 0)
        assert "3,000 statute miles" in table.note and "US carriers only" in table.note


def test_production_code_has_no_airport_exceptions():
    codes = known_codes()
    for name in PRODUCTION:
        for node in ast.walk(ast.parse(source(name))):
            if isinstance(node, ast.Compare):
                for side in (node.left, *node.comparators):
                    constant = isinstance(side, ast.Constant) and isinstance(side.value, str)
                    assert not (constant and side.value in codes), f"{name}:{node.lineno} compares with {side.value}"
            if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                listed = [item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
                assert not (len(listed) > 1 and set(listed) <= codes), f"{name}:{node.lineno} lists airports {listed}"


def test_guidance_names_no_airport_region_count_or_assignment_wording():
    guidance = agent.SYSTEM_PROMPT
    assert not (set(re.findall(r"\b[A-Z0-9]{3}\b", guidance)) & known_codes())
    for phrase in ("New England", "strong candidate", "terminal expansion", "229", "35%", "11 weighted", "usually about"):
        assert phrase not in guidance


def test_regions_appear_only_as_reference_in_tool_descriptions():
    assert "New England" not in source("src/agent.py") + source("src/presentation.py") + source("src/scoring.py")
    assert "New England" not in source("src/tools.py").replace("New England is CT, MA, ME, NH, RI, VT", "")


def test_example_questions_are_only_display_text():
    tree = ast.parse(source("app.py"))
    examples = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EXAMPLES"
    )
    assert len(examples) == 4
    for question in examples:
        assert source("app.py").count(question) == 1
        for name in PRODUCTION[1:]:
            assert question not in source(name)
    assert source("app.py").count("EXAMPLES") == 2


def test_no_code_reads_the_words_of_the_question():
    for name in ("app.py", "src/agent.py", "src/gemini_agent.py", "src/grounding.py", "src/presentation.py"):
        text = source(name)
        assert "question.lower" not in text and "question.casefold" not in text
        assert " in question" not in text
        assert "re.search(question" not in text


def test_question_is_read_only_to_find_catalog_airports_and_never_picks_a_tool():
    """The one exception to the rule above: finding airports decides what must be covered, not how."""
    text = source("src/grounding.py")
    for tool in tools.TOOL_DEFINITIONS:
        assert tool["name"] not in text
    for name in ("app.py", "src/agent.py", "src/presentation.py", "src/gemini_agent.py"):
        assert "_coded_airports" not in source(name) and "_named_airports" not in source(name)
    agent_source = source("src/gemini_agent.py")
    assert "tool_choice" not in agent_source and "allowed_function_names" not in agent_source
