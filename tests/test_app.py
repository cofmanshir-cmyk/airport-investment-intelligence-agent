"""UI checks with real Streamlit and the real Google SDK over a fake network.

No request leaves this machine and no API allowance is used.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import types
from streamlit.testing.v1 import AppTest

from src import config, gemini_agent, presentation
from src.agent import ToolCall
from src import tools

pytestmark = pytest.mark.skipif(
    not (config.PROCESSED_DIR / "ontime_airport_month.parquet").exists(),
    reason="Processed data not built",
)

FAKE_KEY = "fake-key-for-tests-123456"
APP_PATH = str(config.ROOT / "app.py")
SCORE_LABEL = presentation.SCORE_LABEL


def function_call(name, args):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"id": "call_1", "name": name, "args": args}}
    ]}, "finishReason": "STOP"}]}


def text_reply(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}


class FakeNetwork:
    def __init__(self, *replies, status=200):
        self.replies = list(replies)
        self.status = status
        self.generate_calls = 0
        self.keys_seen = []

    def __call__(self, request):
        self.keys_seen.append(request.headers.get("x-goog-api-key"))
        if request.method == "GET":
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            ]})
        self.generate_calls += 1
        return httpx.Response(self.status, json=self.replies.pop(0))


class RecordingNetwork(FakeNetwork):
    """Also keep each request body, to check what the model receives."""

    def __init__(self, *replies):
        super().__init__(*replies)
        self.bodies = []

    def __call__(self, request):
        if request.method != "GET":
            self.bodies.append(json.loads(request.content))
        return super().__call__(request)


@pytest.fixture
def run_app(monkeypatch):
    """Start the page with every Gemini client routed to the fake network."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def start(network):
        def fake_client(api_key):
            return genai.Client(api_key=api_key, http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(network)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ))

        monkeypatch.setattr(gemini_agent, "build_client", fake_client)
        app = AppTest.from_file(APP_PATH, default_timeout=60)
        app.run()
        assert not app.exception
        return app

    return start


def button(app, label):
    return next(item for item in app.button if item.label == label)


def connect(app, model=None):
    app.text_input[0].set_value(FAKE_KEY)
    if model:
        next(item for item in app.selectbox if item.label == "Model").set_value(model)
    button(app, "Connect").click().run()
    assert not app.exception
    return app


def ask(app, question):
    app.text_area[0].set_value(question)
    button(app, "Ask").click().run()
    assert not app.exception
    return app


def page_text(app):
    parts = [item.value for group in (app.markdown, app.caption, app.info, app.warning, app.error, app.success) for item in group]
    return "\n".join(str(part) for part in parts)


def evidence_expanders(app):
    return [item for item in app.expander if item.label.startswith("Evidence")]


def score_expanders(block):
    return [item for item in block.expander if item.label == "How the Airport Opportunity Score works"]


def assert_no_score_ui(app):
    assert not app.info
    assert not score_expanders(app)
    assert tools.SCORE_DISCLAIMER not in page_text(app)


def assert_score_ui_once(app, answers=1):
    assert len(app.info) == answers
    assert all("preliminary screening proxy" in item.value for item in app.info)
    assert len(score_expanders(app)) == answers
    assert not any(item.proto.expanded for item in score_expanders(app))
    assert not [item.value for item in app.warning if item.value.startswith("Preliminary screening proxy")]
    assert presentation.SCORE_PROXY_NOTE not in page_text(app)


def test_page_loads_without_score_ui_and_makes_no_network_call(run_app):
    network = FakeNetwork()
    app = run_app(network)
    assert_no_score_ui(app)
    assert not list(app.expander)
    assert not list(app.dataframe)
    assert network.keys_seen == []


def test_score_answer_explains_the_score_beneath_the_answer(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        text_reply("ILG scores highest.\n\nInsight one.\n\nWarning: ILG has a small base."),
    )
    app = ask(connect(run_app(network)), "Top scores?")
    assert_score_ui_once(app)
    assistant = app.chat_message[1]
    assert len(assistant.info) == 1 and len(score_expanders(assistant)) == 1
    children = list(assistant.children.values())
    position = lambda wanted: next(index for index, child in enumerate(children) if child is wanted)
    insight = next(child for child in assistant.markdown if child.value.startswith("Insight one."))
    assert position(insight) < position(assistant.info[0]) < position(score_expanders(assistant)[0])
    assert position(score_expanders(assistant)[0]) == len(children) - 1
    details = page_text(app)
    assert "35% of the score uses cargo-inclusive departure metrics" in details
    assert "favors larger airports" in details
    assert "2022 base year was still in recovery" in details
    assert "points out of 100, not probabilities" in details
    assert evidence_expanders(app)


def test_methodology_lists_all_11_weights(run_app):
    network = FakeNetwork(function_call("rank_opportunity_score", {"states": [], "top_n": 1}), text_reply("ILG leads."))
    app = ask(connect(run_app(network)), "Top score?")
    table = next(frame.value for frame in app.dataframe if "Weight (%)" in frame.value.columns)
    assert len(table) == 11
    assert table["Weight (%)"].sum() == 100
    weights = dict(zip(table["Measure"], table["Weight (%)"]))
    assert weights["Absolute Passenger Growth 3Y (passengers)"] == 10
    assert weights["Passenger CAGR 3Y (%)"] == 15
    assert weights["Total Load Factor 2025 (%), passengers / seats"] == 15


def test_five_example_types_are_offered(run_app):
    labels = " ".join(item.label for item in run_app(FakeNetwork()).button)
    for example in ("Airport Opportunity Score", "Airport ranking", "Airport comparison", "Airport profile", "Long-haul analysis"):
        assert example in labels


def test_example_only_fills_the_box_and_sends_nothing(run_app):
    network = FakeNetwork()
    app = connect(run_app(network))
    next(item for item in app.button if item.label.startswith("Airport comparison")).click().run()
    assert app.text_area[0].value == "Compare LAX and SNA congestion."
    assert network.generate_calls == 0


def test_ask_is_disabled_until_connected(run_app):
    app = run_app(FakeNetwork())
    assert button(app, "Ask").disabled
    assert not button(connect(app), "Ask").disabled


def test_connecting_lists_models_without_generating_content(run_app):
    network = FakeNetwork()
    app = connect(run_app(network))
    assert network.generate_calls == 0
    assert "Connected to gemini-3.8-flash" in app.success[0].value


def test_a_specific_model_can_be_chosen(run_app):
    app = connect(run_app(FakeNetwork()), model="gemini-3.6-flash")
    assert "Connected to gemini-3.6-flash" in app.success[0].value


def test_unavailable_model_is_reported(run_app):
    app = connect(run_app(FakeNetwork()), model="gemini-3.1-flash-lite")
    assert "not available" in app.error[0].value
    assert not app.session_state["agent"]


def test_empty_key_is_rejected_without_a_network_call(run_app):
    network = FakeNetwork()
    app = run_app(network)
    button(app, "Connect").click().run()
    assert "Enter a Gemini API key" in app.error[0].value
    assert network.keys_seen == []


def test_key_is_used_but_never_shown_or_stored_as_text(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(connect(run_app(network)), "Hello")
    assert FAKE_KEY in network.keys_seen
    assert FAKE_KEY not in page_text(app)
    assert not list(app.text_input)
    stored = app.session_state._state.filtered_state
    assert "agent" in stored
    assert FAKE_KEY not in repr(stored)


def test_comparison_shows_answer_and_python_evidence(run_app):
    network = FakeNetwork(
        function_call("compare_airports", {"airports": ["LAX", "SNA"]}),
        text_reply("SNA had 19.2% of departures delayed and LAX had 18.1%."),
    )
    app = ask(connect(run_app(network)), "Compare LAX and SNA congestion")

    assert network.generate_calls == 2
    assert "SNA had 19.2%" in page_text(app)
    evidence = evidence_expanders(app)
    assert evidence[0].label.startswith("Evidence, methodology, sources and caveats")
    assert not evidence[0].proto.expanded
    assert [tab.label for tab in app.tabs[:5]] == ["Results", "Definitions", "Sources", "Caveats", "Raw"]
    assert not app.warning or "do not match" not in page_text(app)
    congestion = next(frame.value for frame in app.dataframe if "Average taxi-out (min)" in set(frame.value["Measure"]))
    row = congestion[congestion["Measure"] == "Departures delayed 15+ min (%)"].iloc[0]
    assert (row["LAX"], row["SNA"]) == (18.1, 19.2)
    assert "On-Time Performance" in page_text(app)
    assert "PROVISIONAL" in page_text(app)
    assert any("2 of 6" in str(metric.value) for metric in app.metric)


def test_all_four_tools_render(run_app):
    network = FakeNetwork(
        function_call("rank_airports", {"window_years": 3, "states": ["MA"], "top_n": 3, "min_enplanements": 100000}),
        text_reply("Ranking."),
        function_call("get_airport_profile", {"airport": "SFO"}),
        text_reply("Profile."),
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}),
        text_reply("Long haul."),
    )
    app = connect(run_app(network))
    app.session_state["agent"].call_limit = 10
    ask(app, "Rank")
    ask(app, "Profile")
    ask(app, "Long haul")
    columns = [set(frame.value.columns) for frame in app.dataframe]
    assert any("National rank" in item for item in columns)
    assert any("SFO" in item for item in columns)
    assert any("Share at 3000+ miles (%)" in item for item in columns)
    assert len(evidence_expanders(app)) == 3


def test_score_ranking_shows_context_warnings_and_disclaimers(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 25}),
        text_reply("ILG scored 78.38."),
    )
    app = ask(connect(run_app(network)), "Highest scores?")
    table = next(frame.value for frame in app.dataframe if SCORE_LABEL in frame.value.columns)
    first = table.iloc[0]
    assert first["Airport"] == "ILG"
    assert first[SCORE_LABEL] == 78.38
    assert first["National rank"] == "#1 of 229"
    assert presentation.POSITION_LABEL not in table.columns
    assert first["Passengers 2022"] == 4988
    assert first["Passengers 2025"] == first["Passengers 2022"] + first["Absolute Passenger Growth 3Y"]
    assert first["Passenger CAGR 3Y (%)"] == 223.74
    assert first["Base-year warning"] == "Small base"
    assert table[table["Airport"] == "PAE"].iloc[0]["Base-year warning"] == "Small base"
    warnings = " ".join(item.value for item in app.warning)
    assert "ILG: Small base year: 2022 had 4,988 passengers" in warnings
    assert "PAE: Small base year: 2015 had 92 passengers" in warnings
    assert "Preliminary screening proxy" in page_text(app)
    assert any("Absolute Passenger Growth 3Y (passengers)" in set(frame.value.get("Measure", [])) for frame in app.dataframe)


def test_small_base_notice_appears_when_the_model_omits_it(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ILG"}), text_reply("ILG grew quickly."),
    )
    app = ask(connect(run_app(network)), "ILG?")
    notices = [item.value for item in app.warning if item.value.startswith("Small base year: ILG (")]
    assert len(notices) == 1
    assert "2022: 4,988" in notices[0]


def test_small_base_notice_is_not_repeated_when_the_model_states_it(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ILG"}),
        text_reply("ILG grew quickly.\n\nWarning: ILG has a small base in 2015 and 2022."),
    )
    app = ask(connect(run_app(network)), "ILG?")
    assert not [item.value for item in app.warning if item.value.startswith("Small base year: ILG (")]


def test_naming_an_airport_without_the_warning_still_shows_the_notice(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ILG"}),
        text_reply("ILG grew quickly.\n\nWarning: terminal capacity is not directly measured."),
    )
    app = ask(connect(run_app(network)), "ILG?")
    assert [item.value for item in app.warning if item.value.startswith("Small base year: ILG (")]


def test_filtered_score_ranking_keeps_the_national_rank_and_labels_the_position(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": ["CA"], "top_n": 5, "min_passengers_2025": 2000000, "passenger_comparison": "over"}),
        text_reply("SFO leads."),
    )
    app = ask(connect(run_app(network)), "California airports over 2 million?")
    table = next(frame.value for frame in app.dataframe if SCORE_LABEL in frame.value.columns)
    assert table.iloc[0]["Airport"] == "SFO"
    assert table.iloc[0]["National rank"] == "#17 of 229"
    assert table.iloc[0][SCORE_LABEL] == 62.39
    assert table.iloc[0][presentation.POSITION_LABEL] == 1
    text = page_text(app)
    assert "Over 2,000,000 2025 departing passengers" in text
    assert "Filters do not change scores or national ranks" in text


def test_passenger_preset_is_passed_to_the_model_and_the_question_stays_clean(run_app):
    network = RecordingNetwork(text_reply("Answer."))
    app = connect(run_app(network))
    next(item for item in app.selectbox if "Minimum 2025 departing passengers" in item.label).set_value("At least 1 million")
    ask(app, "Top airports?")
    sent = json.dumps(network.bodies[0]["contents"])
    assert "at least 1,000,000 2025 departing passengers" in sent
    assert app.session_state["history"][0][0] == "Top airports?"


def test_default_preset_adds_no_filter_to_the_question(run_app):
    network = RecordingNetwork(text_reply("Answer."))
    ask(connect(run_app(network)), "Which airports should we investigate for terminal expansion?")
    assert "Selected filter" not in json.dumps(network.bodies[0]["contents"])


def test_presets_offer_the_agreed_options(run_app):
    box = next(item for item in run_app(FakeNetwork()).selectbox if "Minimum 2025 departing passengers" in item.label)
    assert box.options == ["All eligible airports", "At least 500,000", "At least 1 million", "At least 5 million", "At least 10 million"]
    assert box.value == "All eligible airports"


def test_answer_separates_conclusion_and_insights_around_the_python_table(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        text_reply("ILG scores highest.\n\n| Airport | Score |\n|---|---|\n| ILG | 99 |\n\nInsight one.\n\nWarning: ILG has a small base."),
    )
    app = ask(connect(run_app(network)), "Top scores?")
    values = [item.value for item in app.markdown]
    assert "ILG scores highest." in values
    insight = next(value for value in values if value.startswith("Insight one."))
    assert "99" not in insight and "| Airport" not in " ".join(values)
    assert next(frame.value for frame in app.dataframe if SCORE_LABEL in frame.value.columns).iloc[0][SCORE_LABEL] == 78.38


def test_factual_question_gets_no_forced_table_or_score(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}),
        text_reply("SFO's Total Load Factor in 2025 was 82.26%."),
    )
    app = ask(connect(run_app(network)), "What is SFO's load factor?")
    tree_text = page_text(app)
    assert tree_text.count("SFO's Total Load Factor") == 1
    assert not [item.value for item in app.warning if "Warning" in item.value]
    assert presentation.main_tables(app.session_state["history"][0][1].tool_calls) == []


def test_comparison_table_shows_only_the_chosen_sections(run_app):
    network = FakeNetwork(
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}),
        text_reply("SNA had more delays."),
    )
    app = ask(connect(run_app(network)), "Compare LAX and SNA congestion")
    tables = presentation.main_tables(app.session_state["history"][0][1].tool_calls)
    assert [table.title for table in tables] == ["Operational congestion"]


def test_material_warning_stays_available_in_evidence_when_collapsed(run_app):
    network = FakeNetwork(function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("ILG leads."))
    app = ask(connect(run_app(network)), "Top?")
    text = page_text(app)
    assert "35% of the score uses cargo-inclusive departure metrics" in text
    assert "terminal capacity" in text


def test_profile_table_shows_score_and_rank_when_chosen(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ATL", "table_sections": ["score"]}),
        text_reply("ATL scored 50.66."),
    )
    app = ask(connect(run_app(network)), "ATL?")
    table = next(
        frame.value for frame in app.dataframe
        if SCORE_LABEL in set(frame.value.get("Measure", []))
    ).set_index("Measure")["ATL"]
    assert table[SCORE_LABEL] == "50.66"
    assert table["National rank"] == "#82 of 229"
    assert table["Absolute Passenger Growth 3Y"] == "5,924,745"
    assert table["Passenger CAGR 3Y (%)"] == "4.16"


def test_airport_without_a_score_says_why(run_app):
    network = FakeNetwork(function_call("get_airport_profile", {"airport": "JAC"}), text_reply("JAC has no score."))
    app = ask(connect(run_app(network)), "JAC?")
    assert "JAC has no score: 2022 is incomplete: 11 of 12 months reported." in page_text(app)


def test_absolute_ranking_is_labelled_and_keeps_cagr_visible(run_app):
    network = FakeNetwork(
        function_call("rank_airports", {
            "window_years": 3, "states": [], "top_n": 3, "min_enplanements": 100000,
            "rank_by": "absolute_passenger_growth",
        }),
        text_reply("ORD added the most passengers."),
    )
    app = ask(connect(run_app(network)), "Most passengers added?")
    table = next(frame.value for frame in app.dataframe if "Absolute Passenger Growth 3Y" in frame.value.columns)
    assert table.iloc[0]["Airport"] == "ORD"
    assert table.iloc[0]["Passenger CAGR 3Y (%)"] == 7.11
    text = page_text(app)
    assert "Ranked by Absolute Passenger Growth 3Y" in text
    assert "tends to favor larger airports" in text


def test_newest_answer_appears_first_and_only_it_is_expanded(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}), text_reply("First answer."),
        function_call("get_airport_profile", {"airport": "BOS"}), text_reply("Second answer."),
    )
    app = ask(ask(connect(run_app(network)), "First"), "Second")
    text = page_text(app)
    assert text.index("Second answer.") < text.index("First answer.")
    assert [item.proto.expanded for item in evidence_expanders(app)] == [False, False]


RANK_CAGR = {"window_years": 3, "states": [], "top_n": 3, "min_enplanements": 100000}
ATL_PROFILE = {"airport": "ATL", "table_sections": []}


@pytest.mark.parametrize(
    "replies",
    [
        [function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("ILG leads.")],
        [function_call("rank_opportunity_score", {"states": ["ZZ"], "top_n": 3}),
         function_call("rank_opportunity_score", {"states": ["CA"], "top_n": 3}), text_reply("SFO leads.")],
        [function_call("get_airport_profile", {"airport": "ATL", "table_sections": ["score"]}), text_reply("See the table.")],
        [function_call("get_airport_profile", ATL_PROFILE), text_reply("ATL scored 50.66.")],
        [function_call("get_airport_profile", ATL_PROFILE), text_reply("ATL reached 50.66 points.")],
        [function_call("get_airport_profile", ATL_PROFILE), text_reply("ATL is #82 of 229 nationally.")],
        [function_call("get_airport_profile", {"airport": "JAC", "table_sections": ["score"]}), text_reply("See the table.")],
        [function_call("compare_airports", {"airports": ["LAX", "SNA"]}), text_reply("Compared.")],
        [function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["score", "congestion"]}), text_reply("Compared.")],
        [function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
         function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["score"]}), text_reply("Both.")],
    ],
    ids=[
        "score_ranking", "score_tool_rejected_then_used", "profile_score_table", "profile_narrative_word",
        "profile_narrative_value", "profile_narrative_rank", "unscored_airport_table", "comparison_default_table",
        "comparison_score_section", "two_scoring_tools",
    ],
)
def test_answers_reporting_the_score_show_its_ui_once(run_app, replies):
    # The question never mentions the score, so only the tools and the answer can trigger this.
    app = ask(connect(run_app(FakeNetwork(*replies))), "Tell me about these airports")
    assert_score_ui_once(app)


@pytest.mark.parametrize(
    "replies",
    [
        [function_call("rank_airports", RANK_CAGR), text_reply("These are raw values, not the Airport Opportunity Score.")],
        [function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply("ANC is 2.6% long-haul.")],
        [function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}), text_reply("SFO grew 8.87% a year.")],
        [function_call("get_airport_profile", {"airport": "SFO", "table_sections": ["demand"]}), text_reply("SFO grew 8.87% a year.")],
        [function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}), text_reply("SNA had more delays.")],
        [text_reply("Hello.")],
        [function_call("rank_opportunity_score", {"states": ["ZZ"], "top_n": 3}), text_reply("That state code was rejected.")],
        [function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}), text_reply("SFO's CAGR is one input to the score.")],
        [function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}), text_reply("Warning: 35% of the score uses cargo-inclusive metrics.")],
        [function_call("get_airport_profile", {"airport": "ATL", "table_sections": []}), text_reply("Some share was 50.66% and ATL ranks #82 by size.")],
        [function_call("get_airport_profile", {"airport": "JAC", "table_sections": []}), text_reply("JAC has no score.")],
        [text_reply("I can explain the Opportunity Score.")],
    ],
    ids=[
        "cagr_ranking", "long_haul", "profile_fact", "profile_demand_table", "congestion", "no_tool", "rejected_score_tool",
        "incidental_score_word", "score_word_in_warning", "same_digits_as_percentage_and_bare_rank", "no_score_value_to_show",
        "score_word_without_a_tool",
    ],
)
def test_answers_not_reporting_the_score_show_no_score_ui(run_app, replies):
    # The question names the score, so a keyword check on the question would fail this test.
    app = ask(connect(run_app(FakeNetwork(*replies))), "What is the Airport Opportunity Score here?")
    assert_no_score_ui(app)


@pytest.mark.parametrize(
    "follow_up",
    [
        [text_reply("ATL scored 50.66 in the earlier result.")],
        [function_call("rank_airports", RANK_CAGR), text_reply("ATL still has its 50.66 from before.")],
    ],
    ids=["no_tool", "unrelated_tool"],
)
def test_follow_up_restating_an_earlier_score_shows_the_score_ui(run_app, follow_up):
    network = FakeNetwork(function_call("get_airport_profile", ATL_PROFILE), text_reply("ATL grew."), *follow_up)
    app = ask(ask(connect(run_app(network)), "ATL?"), "And its score?")
    assert [len(score_expanders(app.chat_message[index])) for index in (1, 3)] == [1, 0]


def test_follow_up_with_an_unrelated_rank_shows_no_score_ui(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 5}), text_reply("ILG leads."),
        function_call("rank_airports", RANK_CAGR), text_reply("ORD is #3 by passenger CAGR."),
    )
    app = ask(ask(connect(run_app(network)), "Scores?"), "Growth?")
    assert [len(score_expanders(app.chat_message[index])) for index in (1, 3)] == [0, 1]


def test_displayed_score_table_has_one_disclaimer_and_keeps_its_no_score_reason(run_app):
    network = FakeNetwork(
        function_call("compare_airports", {"airports": ["ATL", "JAC"], "table_sections": ["score"]}), text_reply("Compared."),
    )
    app = ask(connect(run_app(network)), "ATL and JAC?")
    assert_score_ui_once(app)
    assert "JAC has no score: 2022 is incomplete: 11 of 12 months reported." in [item.value for item in app.chat_message[1].caption]
    assert any(SCORE_LABEL in set(frame.value.get("Measure", [])) for frame in app.dataframe)
    assert "PROVISIONAL" in page_text(app)


def test_score_table_keeps_its_caption_in_evidence_when_the_answer_shows_no_disclaimer(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}), text_reply("SFO grew 8.87% a year."),
    )
    app = ask(connect(run_app(network)), "SFO growth?")
    assert_no_score_ui(app)
    assert [item.value for item in app.caption].count(presentation.SCORE_PROXY_NOTE) == 1


def test_command_line_tables_keep_the_score_caption():
    result = tools.get_airport_profile("ATL", ["score"])
    call = ToolCall("get_airport_profile", {"airport": "ATL", "table_sections": ["score"]}, result, False)
    assert presentation.main_tables([call])[0].note == presentation.SCORE_PROXY_NOTE


def test_each_answer_decides_for_itself(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1}), text_reply("ILG leads."),
        function_call("rank_airports", RANK_CAGR), text_reply("Fastest growth."),
        function_call("rank_opportunity_score", {"states": ["CA"], "top_n": 1}), text_reply("SFO leads."),
    )
    app = connect(run_app(network))
    app.session_state["agent"].call_limit = 10
    ask(ask(ask(app, "First"), "Second"), "Third")
    assert_score_ui_once(app, answers=2)
    # Newest first: the third answer, then the second, then the first.
    assert [len(score_expanders(app.chat_message[index])) for index in (1, 3, 5)] == [1, 0, 1]


def test_answers_without_the_score_keep_their_scope_and_caveats(run_app):
    network = FakeNetwork(
        function_call("rank_airports", RANK_CAGR), text_reply("Fastest growth.\n\nWarning: 2022 was a recovery year."),
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply("ANC is 2.6% long-haul."),
    )
    app = ask(ask(connect(run_app(network)), "Growth?"), "Long haul?")
    text = page_text(app)
    assert "Warning: 2022 was a recovery year." in text
    assert "One raw metric, not an investment ranking" in text
    assert "Full calendar years 2022 and 2025" in text
    assert "T-100 passengers include people connecting" in text
    assert "there is no single standard definition" in text
    assert "The route file covers US carriers only." in text
    assert "The threshold is an assumption" in text
    assert_no_score_ui(app)


def test_evidence_stays_complete_when_the_score_is_not_reported(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO", "table_sections": []}), text_reply("SFO grew 8.87% a year."),
    )
    app = ask(connect(run_app(network)), "SFO growth?")
    assert_no_score_ui(app)
    assert len(evidence_expanders(app)) == 1
    assert any(SCORE_LABEL in set(frame.value.get("Measure", [])) for frame in app.dataframe)
    assert "PROVISIONAL" in page_text(app)
    assert "SFO annual inputs" in "\n".join(item.value for item in app.markdown)
    call = app.session_state["history"][0][1].tool_calls[0]
    assert presentation.build_evidence(call).raw["disclaimer"] == tools.SCORE_DISCLAIMER


def test_invented_number_warning_is_shown(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}),
        text_reply("SFO has 1,234,567 unmet passengers."),
    )
    app = ask(connect(run_app(network)), "Unmet demand at SFO?")
    assert "1,234,567" in " ".join(item.value for item in app.warning)


def test_answer_without_a_tool_is_flagged(run_app):
    app = ask(connect(run_app(FakeNetwork(text_reply("Hello.")))), "Hello")
    assert any("No tool was called" in item.value for item in app.warning)


def test_server_error_is_shown_once_and_never_retried(run_app):
    network = FakeNetwork({"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}, status=503)
    app = ask(connect(run_app(network)), "Question")
    assert network.generate_calls == 1
    assert "503" in app.error[0].value and "try again later" in app.error[0].value
    app.run()
    assert network.generate_calls == 1


class StalledNetwork(FakeNetwork):
    """List models normally, then time out on every question."""

    def __call__(self, request):
        if request.method == "GET":
            return super().__call__(request)
        self.generate_calls += 1
        raise httpx.ReadTimeout("timed out", request=request)


def test_timeout_is_shown_as_an_error_and_never_resent(run_app):
    network = StalledNetwork()
    app = ask(connect(run_app(network)), "Question")
    assert "did not respond in time" in app.error[0].value
    assert network.generate_calls == 1
    assert app.session_state["history"] == []
    app.run()
    assert not app.exception
    assert network.generate_calls == 1


def test_timeout_while_connecting_is_shown_as_an_error(run_app):
    def stalled(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    app = connect(run_app(stalled))
    assert "did not respond in time" in app.error[0].value
    assert not app.session_state["agent"]


def test_rerun_does_not_send_the_question_again(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(connect(run_app(network)), "Hello")
    app.run()
    app.run()
    assert network.generate_calls == 1


def test_blank_question_sends_nothing(run_app):
    network = FakeNetwork()
    app = ask(connect(run_app(network)), "   ")
    assert network.generate_calls == 0


def test_session_limit_is_shown_without_a_call(run_app):
    network = FakeNetwork()
    app = connect(run_app(network))
    app.session_state["agent"].calls_used = app.session_state["agent"].call_limit
    ask(app, "Question")
    assert "session limit" in app.error[0].value
    assert network.generate_calls == 0


def test_disconnect_clears_the_connection_and_conversation(run_app):
    app = ask(connect(run_app(FakeNetwork(text_reply("Hello.")))), "Hello")
    button(app, "Disconnect").click().run()
    assert app.session_state["agent"] is None
    assert app.session_state["history"] == []
    assert not evidence_expanders(app)


def test_dollar_signs_are_shown_literally():
    assert presentation.escape_markdown("Costs $5 and $6") == r"Costs \$5 and \$6"


def test_presentation_copies_tool_values_exactly():
    result = tools.compare_airports(["LAX", "SNA"])
    evidence = presentation.build_evidence(ToolCall("compare_airports", {"airports": ["LAX", "SNA"]}, result, False))
    demand = next(table.data for table in evidence.tables if table.title == "Demand-pressure proxies")
    row = demand[demand["Measure"] == "Total Load Factor 2025 (%), passengers / seats"].iloc[0]
    assert row["LAX"] == result["airports"]["LAX"]["demand_pressure_proxies"]["metrics"]["total_load_factor_2025_pct"]
    assert {table.title for table in evidence.tables} == {
        "Airport Opportunity Score", "Score points by metric", "Demand-pressure proxies",
        "Operational congestion", "Cargo context, not scored",
    }
    assert evidence.provisional and all(item.startswith("PROVISIONAL") for item in evidence.provisional)
    assert json.dumps(evidence.raw, default=float)


def test_rejected_tool_input_is_presented_as_an_error():
    evidence = presentation.build_evidence(ToolCall("get_airport_profile", {"airport": "ZZZZ"}, {"error": "Not found"}, True))
    assert evidence.error == "Not found"
    assert evidence.tables == []
