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
        app.selectbox[0].set_value(model)
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


def test_page_loads_with_score_disclaimer_and_makes_no_network_call(run_app):
    network = FakeNetwork()
    app = run_app(network)
    banner = app.info[0].value
    assert "Airport Opportunity Score: Preliminary screening proxy" in banner
    assert "35% of the score uses cargo-inclusive departure metrics" in banner
    assert "favors larger airports" in banner
    assert "2022 base year was still in recovery" in banner
    assert network.keys_seen == []


def test_methodology_lists_all_11_weights(run_app):
    app = run_app(FakeNetwork())
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
    assert evidence[0].label.startswith("Evidence, sources and caveats")
    assert evidence[0].proto.expanded
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
    table = next(frame.value for frame in app.dataframe if "Airport Opportunity Score" in frame.value.columns)
    first = table.iloc[0]
    assert first["Airport"] == "ILG"
    assert first["Airport Opportunity Score"] == 78.38
    assert first["Passengers 2022"] == 4988
    assert first["Passengers 2025"] == first["Passengers 2022"] + first["Absolute Passenger Growth 3Y"]
    assert first["Passenger CAGR 3Y (%)"] == 223.74
    assert first["Base-year warning"] == "Small base"
    assert table[table["Airport"] == "PAE"].iloc[0]["Base-year warning"] == "Small base"
    warnings = " ".join(item.value for item in app.warning)
    assert "ILG: Small base year: 2022 had 4,988 passengers" in warnings
    assert "PAE: Small base year: 2015 had 92 passengers" in warnings
    assert "Preliminary screening proxy" in warnings
    assert any("Absolute Passenger Growth 3Y (passengers)" in set(frame.value.get("Measure", [])) for frame in app.dataframe)


def test_small_base_warning_is_shown_outside_the_collapsed_evidence(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "ILG"}), text_reply("First."),
        function_call("get_airport_profile", {"airport": "SFO"}), text_reply("Second."),
    )
    app = ask(ask(connect(run_app(network)), "ILG?"), "SFO?")
    assert any(item.value.startswith("ILG: Small base year") for item in app.warning)


def test_filtered_score_ranking_keeps_the_national_rank(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": ["DE"], "top_n": 5}),
        text_reply("ILG scored 78.38."),
    )
    app = ask(connect(run_app(network)), "Delaware score?")
    table = next(frame.value for frame in app.dataframe if "Airport Opportunity Score" in frame.value.columns)
    assert table.iloc[0]["National rank"] == 1
    assert table.iloc[0]["Airport Opportunity Score"] == 78.38


def test_profile_shows_the_score_table_with_passenger_context(run_app):
    network = FakeNetwork(function_call("get_airport_profile", {"airport": "ATL"}), text_reply("ATL scored 50.66."))
    app = ask(connect(run_app(network)), "ATL?")
    table = next(
        frame.value for frame in app.dataframe
        if "Airport Opportunity Score" in set(frame.value.get("Measure", []))
    ).set_index("Measure")["ATL"]
    assert table["Airport Opportunity Score"] == 50.66
    assert table["National rank"] == 82
    assert table["Absolute Passenger Growth 3Y"] == 5924745
    assert table["Passenger CAGR 3Y (%)"] == 4.16
    assert table["Passengers 2025"] - table["Passengers 2022"] == 5924745


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
    assert [item.proto.expanded for item in evidence_expanders(app)] == [True, False]


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
