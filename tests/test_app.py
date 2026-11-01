"""UI checks with real Streamlit and the real Google SDK over a fake network.

No request leaves this machine and no API allowance is used.
"""

import json

import httpx
import pandas as pd
import pytest
from google import genai
from google.genai import errors, types
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Caption, Dataframe, Markdown

from src import cli, config, gemini_agent, presentation
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
        self.statuses = []
        self.urls = []

    def __call__(self, request):
        self.keys_seen.append(request.headers.get("x-goog-api-key"))
        if request.method == "GET":
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            ]})
        self.generate_calls += 1
        self.urls.append(str(request.url))
        status = self.statuses.pop(0) if self.statuses else self.status
        return httpx.Response(status, json=self.replies.pop(0))


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
    """Start the page with a server-side key and every Gemini client routed to the fake network."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def start(network, key=FAKE_KEY, secrets=None):
        if key:
            monkeypatch.setenv("GEMINI_API_KEY", key)
        else:
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        def fake_client(api_key):
            return genai.Client(api_key=api_key, http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(network)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ))

        monkeypatch.setattr(gemini_agent, "build_client", fake_client)
        app = AppTest.from_file(APP_PATH, default_timeout=60)
        # Always set here, so tests never read a real secrets file on this machine.
        for name, value in (secrets or {"GEMINI_API_KEY": ""}).items():
            app.secrets[name] = value
        app.run()
        assert not app.exception
        return app

    return start


def button(app, label):
    return next(item for item in app.button if item.label == label)


def connect(app):
    """The page connects itself from the server-side key, so tests only confirm it did."""
    assert app.session_state["agent"]
    return app


def ask(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception
    return app


def direct(block, kind):
    """List what sits directly in a message, leaving out anything inside its expanders."""
    return [child for child in block.children.values() if isinstance(child, kind)]


def user_messages(app):
    return [item for item in app.chat_message if item.name == "user"]


def assistant_messages(app):
    return [item for item in app.chat_message if item.name == "assistant"]


def main_frames(message):
    return [item.value for item in direct(message, Dataframe)]


def card_questions(app):
    return [item.label for item in app.button if item.key and item.key.startswith("example-")]


def page_text(app):
    parts = [item.value for group in (app.markdown, app.caption, app.info, app.warning, app.error, app.success) for item in group]
    return "\n".join(str(part) for part in parts)


def evidence_expanders(app):
    return [item for item in app.expander if item.label == "Explore the data & sources"]


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
    evidence = evidence_expanders(assistant)[0]
    assert position(insight) < position(assistant.info[0]) < position(score_expanders(assistant)[0]) < position(evidence)
    assert position(evidence) == len(children) - 1
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


EXAMPLE_CARDS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]


def sidebar_labels(app):
    return [item.label for group in (app.sidebar.selectbox, app.sidebar.button, app.sidebar.text_input) for item in group]


def test_empty_chat_shows_only_the_four_exact_question_cards(run_app):
    network = FakeNetwork()
    app = run_app(network)
    assert sorted(card_questions(app)) == sorted(EXAMPLE_CARDS)
    for heading in ("Expansion opportunities", "Airport congestion", "Long-haul traffic", "Unmet demand"):
        assert heading not in page_text(app)
    assert not list(app.chat_message)
    assert not list(app.expander)
    assert not list(app.dataframe)
    assert_no_score_ui(app)
    assert not app.chat_input[0].disabled
    assert button(app, "New Chat").disabled
    assert network.generate_calls == 0


def test_cards_fill_two_columns_and_each_card_is_one_full_width_button():
    source = (config.ROOT / "app.py").read_text()
    assert "st.columns(2)" in source
    assert 'width="stretch"' in source
    assert "st.container(border=True)" not in source


def test_cards_are_disabled_with_a_setup_message_when_no_key_is_configured(run_app):
    network = FakeNetwork()
    app = run_app(network, key=None)
    assert "GEMINI_API_KEY" in app.warning[0].value
    assert app.chat_input[0].disabled
    cards = [item for item in app.button if item.key and item.key.startswith("example-")]
    assert len(cards) == 4 and all(item.disabled for item in cards)
    assert network.keys_seen == []
    assert app.session_state["agent"] is None
    assert not list(app.selectbox)


def test_missing_or_malformed_secrets_file_shows_the_setup_message(run_app, monkeypatch):
    def unavailable(self, *args):
        raise StreamlitSecretNotFoundError("No secrets found.")

    monkeypatch.setattr(Secrets, "get", unavailable)
    app = run_app(FakeNetwork(), key=None)
    assert "GEMINI_API_KEY" in app.warning[0].value
    assert app.session_state["agent"] is None


def test_blank_key_counts_as_missing(run_app, monkeypatch):
    app = run_app(FakeNetwork(), key=None, secrets={"GEMINI_API_KEY": "   "})
    assert "GEMINI_API_KEY" in app.warning[0].value


def test_key_is_read_from_the_environment(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(run_app(network), "Hello")
    assert network.keys_seen[0] == FAKE_KEY
    assert not any("GEMINI_API_KEY" in item.value for item in app.warning)


def test_key_is_read_from_streamlit_secrets(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(run_app(network, key=None, secrets={"GEMINI_API_KEY": "secret-file-key-987"}), "Hello")
    assert set(network.keys_seen) == {"secret-file-key-987"}
    assert not any("GEMINI_API_KEY" in item.value for item in app.warning)
    assert "secret-file-key-987" not in page_text(app)


def test_environment_key_is_used_before_secrets(run_app):
    network = FakeNetwork()
    run_app(network, secrets={"GEMINI_API_KEY": "secret-file-key-987"})
    assert set(network.keys_seen) == {FAKE_KEY}


def test_secrets_file_is_ignored_by_git():
    assert ".streamlit/secrets.toml" in (config.ROOT / ".gitignore").read_text().splitlines()


def test_page_asks_for_no_key_and_never_shows_it(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(run_app(network), "Hello")
    assert not list(app.text_input)
    assert FAKE_KEY not in page_text(app)
    assert FAKE_KEY not in str(app.main) and FAKE_KEY not in str(app.sidebar)
    assert FAKE_KEY not in repr(app.session_state._state.filtered_state)


def test_error_message_that_echoes_the_key_is_not_shown(run_app):
    network = FakeNetwork({"error": {"code": 500, "message": f"failed for {FAKE_KEY}", "status": "INTERNAL"}}, status=500)
    app = ask(run_app(network), "Question")
    assert FAKE_KEY not in page_text(app)
    assert "failed for" not in app.error[0].value


class RejectingThenWorkingNetwork(FakeNetwork):
    """Reject the key once when listing models, then behave normally."""

    def __init__(self):
        super().__init__()
        self.rejected = False

    def __call__(self, request):
        if request.method == "GET" and not self.rejected:
            self.rejected = True
            return httpx.Response(403, json={"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}})
        return super().__call__(request)


def test_connection_failure_is_explained_without_crashing_and_can_be_retried(run_app):
    network = RejectingThenWorkingNetwork()
    app = run_app(network)
    assert "Could not connect to Gemini" in app.error[0].value
    assert app.chat_input[0].disabled
    app.run()
    assert len(network.keys_seen) == 0
    button(app, "Try again").click().run()
    assert not app.exception and not app.error
    assert app.session_state["agent"].model == "gemini-3.8-flash"


def test_listing_timeout_is_explained_without_crashing(run_app):
    def stalled(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    app = run_app(stalled)
    assert "did not respond in time" in app.error[0].value
    assert app.session_state["agent"] is None


def test_sidebar_has_only_the_model_selector_and_no_removed_controls(run_app):
    network = FakeNetwork(text_reply("Hello."))
    app = ask(run_app(network), "Hello")
    assert sidebar_labels(app) == ["Model"]
    assert not list(app.metric)
    assert not list(app.success)
    assert "Minimum 2025 departing passengers" not in str(app.main)
    assert "calls this session" not in page_text(app)
    source = (config.ROOT / "app.py").read_text()
    for removed in ("calls_used", "call_limit", "PASSENGER_PRESETS", "Disconnect", "st.metric", 'type="password"'):
        assert removed not in source


def test_sidebar_has_no_data_summary_and_answers_keep_their_sources(run_app):
    network = FakeNetwork(
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply("ANC is 2.6% long-haul."),
    )
    app = ask(run_app(network), "Long haul?")
    assert not list(app.sidebar.markdown) and not list(app.sidebar.caption) and not list(app.sidebar.header)
    assert "About the data" not in page_text(app)
    evidence = evidence_expanders(assistant_messages(app)[0])[0]
    text = "\n".join(item.value for item in evidence.markdown)
    assert tools.SOURCES["t100_routes"] in text
    assert "The route file covers US carriers only." in text


def test_readme_keeps_the_data_sources_removed_from_the_sidebar():
    readme = (config.ROOT / "README.md").read_text()
    for detail in ("BTS T-100, 2015, 2022 and 2025", "BTS On-Time Performance, 2025", "Jan-Jul 2026, US carriers only", "FAA CY2025 enplanements"):
        assert detail in readme


def test_model_selector_offers_automatic_and_only_available_models_without_generating(run_app):
    network = FakeNetwork()
    app = run_app(network)
    box = next(item for item in app.selectbox if item.label == "Model")
    assert box.options == ["Automatic (newest available)", "gemini-3.8-flash", "gemini-3.6-flash"]
    assert box.value == "Automatic (newest available)"
    assert app.session_state["agent"].model == "gemini-3.8-flash"
    assert network.generate_calls == 0


def test_models_are_listed_once_across_reruns(run_app):
    network = FakeNetwork()
    app = run_app(network)
    listed = len(network.keys_seen)
    app.run()
    app.run()
    assert len(network.keys_seen) == listed == 1


def test_choosing_a_model_uses_it_and_starts_a_new_chat(run_app):
    network = RecordingNetwork(text_reply("First."), text_reply("Second."))
    app = ask(run_app(network), "First")
    next(item for item in app.selectbox if item.label == "Model").set_value("gemini-3.6-flash").run()
    assert not app.exception
    assert app.session_state["agent"].model == "gemini-3.6-flash"
    assert app.session_state["history"] == []
    assert sorted(card_questions(app)) == sorted(EXAMPLE_CARDS)
    ask(app, "Second")
    assert [item["role"] for item in network.bodies[-1]["contents"]] == ["user"]


def test_configured_model_is_preselected_and_used(run_app, monkeypatch):
    network = FakeNetwork()
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(gemini_agent, "build_client", lambda api_key: genai.Client(api_key=api_key, http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(network)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )))
    app.run()
    assert next(item for item in app.selectbox if item.label == "Model").value == "gemini-3.6-flash"
    assert app.session_state["agent"].model == "gemini-3.6-flash"


def test_no_session_call_cap_across_many_questions(run_app):
    questions = 10
    network = FakeNetwork(*[text_reply(f"Answer {position}.") for position in range(questions)])
    app = run_app(network)
    for position in range(questions):
        ask(app, f"Question {position}")
    assert not app.error
    assert network.generate_calls == questions
    assert len(app.session_state["history"]) == questions


def quota_error(*violations):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": list(violations)}] if violations else []
    return {"error": {"code": 429, "message": "RAW GOOGLE TEXT", "status": "RESOURCE_EXHAUSTED", "details": details}}


PER_MODEL = {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaDimensions": {"model": "gemini-3.8-flash"}}
PER_MODEL_BY_ID = {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}
KEY_WIDE = {"quotaId": "GenerateRequestsPerDayPerProject", "quotaDimensions": {"location": "global"}}


@pytest.mark.parametrize(
    "violations, expected, unexpected",
    [
        ((PER_MODEL,), "selected model may have reached its usage limit", "may not help"),
        ((PER_MODEL_BY_ID,), "selected model may have reached its usage limit", "may not help"),
        ((KEY_WIDE,), "switching models may not help", "Try another available model"),
        ((PER_MODEL, KEY_WIDE), "switching models may not help", "Try another available model"),
        ((), "unless the limit applies to the whole API key", "selected model may have"),
    ],
    ids=["model_dimension", "model_in_id", "key_wide", "mixed", "unspecified"],
)
def test_quota_message_matches_the_scope_of_the_limit(run_app, violations, expected, unexpected):
    network = FakeNetwork(quota_error(*violations), status=429)
    app = ask(run_app(network), "Question")
    message = app.error[0].value
    assert expected in message
    assert unexpected not in message
    assert "RAW GOOGLE TEXT" not in page_text(app)
    assert network.generate_calls == 1
    app.run()
    assert network.generate_calls == 1


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_overload_suggests_waiting_or_another_model(run_app, status_code):
    network = FakeNetwork({"error": {"code": status_code, "message": "RAW GOOGLE TEXT", "status": "UNAVAILABLE"}}, status=status_code)
    app = ask(run_app(network), "Question")
    message = app.error[0].value
    assert "temporarily overloaded or unavailable" in message
    assert "another available model" in message
    assert "RAW GOOGLE TEXT" not in page_text(app)
    assert network.generate_calls == 1


@pytest.mark.parametrize(
    "status_code, body",
    [
        (403, {"code": 403, "message": "RAW GOOGLE TEXT", "status": "PERMISSION_DENIED"}),
        (401, {"code": 401, "message": "RAW GOOGLE TEXT", "status": "UNAUTHENTICATED"}),
        (400, {"code": 400, "message": "API_KEY_INVALID RAW GOOGLE TEXT", "status": "INVALID_ARGUMENT"}),
        (400, {"code": 400, "message": "RAW GOOGLE TEXT", "status": "FAILED_PRECONDITION"}),
    ],
    ids=["forbidden", "unauthenticated", "invalid_key", "failed_precondition"],
)
def test_key_or_setup_problems_do_not_suggest_switching_models(run_app, status_code, body):
    app = ask(run_app(FakeNetwork({"error": body}, status=status_code)), "Question")
    message = app.error[0].value
    assert "Switching models won't help" in message
    assert "Try another" not in message
    assert "RAW GOOGLE TEXT" not in page_text(app)


def test_other_errors_show_a_friendly_message_without_raw_text(run_app):
    network = FakeNetwork({"error": {"code": 418, "message": "RAW GOOGLE TEXT", "status": "ODD"}}, status=418)
    app = ask(run_app(network), "Question")
    assert "could not complete this request" in app.error[0].value
    assert "RAW GOOGLE TEXT" not in page_text(app)
    assert "418" not in app.error[0].value


def test_unusual_error_details_do_not_break_the_message():
    for details in (None, [], "text", {"error": "text"}, {"error": {"details": "text"}}, {"error": {"details": ["text", {"violations": "text"}]}}):
        error = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}, None)
        error.details = details
        assert "usage limit" in cli._gemini_error(error)


def test_failed_question_stays_visible_and_is_not_resent_until_asked(run_app):
    network = FakeNetwork(quota_error(PER_MODEL), text_reply("Now answered."))
    network.statuses = [429, 200]
    app = ask(run_app(network), "My exact question")
    assert app.session_state["failed_question"] == "My exact question"
    assert [item.markdown[0].value for item in user_messages(app)] == ["My exact question"]
    assert card_questions(app) == []
    assert not assistant_messages(app)
    app.run()
    assert network.generate_calls == 1

    button(app, "Ask again").click().run()
    assert not app.exception
    assert network.generate_calls == 2
    assert [item.markdown[0].value for item in user_messages(app)] == ["My exact question"]
    assert [item.markdown[0].value for item in assistant_messages(app)] == ["Now answered."]
    assert not app.error
    assert app.session_state["failed_question"] is None
    assert not [item for item in app.button if item.label == "Ask again"]


def test_failed_question_survives_a_model_change_without_being_sent(run_app):
    network = RecordingNetwork(quota_error(PER_MODEL), text_reply("Answered by the other model."))
    network.statuses = [429, 200]
    app = ask(run_app(network), "My exact question")
    next(item for item in app.selectbox if item.label == "Model").set_value("gemini-3.6-flash").run()
    assert not app.exception
    assert network.generate_calls == 1
    assert not app.error
    assert [item.markdown[0].value for item in user_messages(app)] == ["My exact question"]

    button(app, "Ask again").click().run()
    assert network.generate_calls == 2
    assert "gemini-3.6-flash" in network.urls[-1]
    assert [item["role"] for item in network.bodies[-1]["contents"]] == ["user"]
    assert network.bodies[-1]["contents"][0]["parts"][0]["text"] == "My exact question"


def test_typing_a_new_question_replaces_the_failed_one(run_app):
    network = FakeNetwork(quota_error(PER_MODEL), text_reply("Answered."))
    network.statuses = [429, 200]
    app = ask(ask(run_app(network), "Old question"), "New question")
    assert network.generate_calls == 2
    assert [item.markdown[0].value for item in user_messages(app)] == ["New question"]
    assert app.session_state["failed_question"] is None


def test_failed_follow_up_keeps_earlier_answers_and_comes_last(run_app):
    network = FakeNetwork(text_reply("First answer."), quota_error(PER_MODEL))
    network.statuses = [200, 429]
    app = ask(ask(run_app(network), "First"), "Second")
    assert [item.markdown[0].value for item in user_messages(app)] == ["First", "Second"]
    assert [item.name for item in app.chat_message] == ["user", "assistant", "user"]
    assert app.error


def test_new_chat_clears_a_failed_question(run_app):
    app = ask(run_app(FakeNetwork(quota_error(PER_MODEL), status=429)), "Question")
    button(app, "New Chat").click().run()
    assert app.session_state["failed_question"] is None
    assert not list(app.chat_message) and not app.error
    assert sorted(card_questions(app)) == sorted(EXAMPLE_CARDS)
    assert not [item for item in app.button if item.label == "Ask again"]


def test_timeout_also_keeps_the_question_for_asking_again(run_app):
    network = StalledNetwork()
    app = ask(run_app(network), "Question")
    assert "did not respond in time" in app.error[0].value
    assert "connect with" not in app.error[0].value
    assert [item.markdown[0].value for item in user_messages(app)] == ["Question"]
    assert [item for item in app.button if item.label == "Ask again"]


def test_tables_use_the_native_scrolling_table_and_the_only_style_is_scoped_to_the_cards():
    source = (config.ROOT / "app.py").read_text()
    assert "st.dataframe(" in source
    assert "st.table(" not in source
    assert "unsafe_allow_html" not in source
    assert source.count("st.html(") == 1
    assert source.count(".st-key-example-cards") == 2


@pytest.mark.parametrize("question", EXAMPLE_CARDS)
def test_card_click_sends_its_exact_question_once_through_the_agent(run_app, question):
    network = RecordingNetwork(text_reply("A reply that only the agent could return."))
    app = connect(run_app(network))
    next(item for item in app.button if item.label == question).click().run()
    assert not app.exception
    assert network.generate_calls == 1
    assert network.bodies[0]["contents"][-1]["parts"][0]["text"] == question
    assert app.session_state["history"][0][0] == question
    assert [item.markdown[0].value for item in user_messages(app)] == [question]
    assert assistant_messages(app)[0].markdown[0].value == "A reply that only the agent could return."
    assert card_questions(app) == []
    app.run()
    assert network.generate_calls == 1
    assert len(user_messages(app)) == 1


def test_first_question_shows_one_question_then_one_answer(run_app):
    network = FakeNetwork(text_reply("Hello there."))
    app = ask(connect(run_app(network)), "Hello")
    assert [item.name for item in app.chat_message] == ["user", "assistant"]
    assert app.chat_message[0].markdown[0].value == "Hello"
    assert app.chat_message[1].markdown[0].value == "Hello there."
    assert card_questions(app) == []
    assert not button(app, "New Chat").disabled


def test_turns_stay_in_order_and_keep_the_agent_context(run_app):
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("First answer."),
        text_reply("Second answer."),
        text_reply("Third answer."),
    )
    app = connect(run_app(network))
    agent = app.session_state["agent"]
    ask(ask(ask(app, "First"), "Second"), "Third")
    assert [item.name for item in app.chat_message] == ["user", "assistant"] * 3
    assert [item.markdown[0].value for item in user_messages(app)] == ["First", "Second", "Third"]
    assert [item.markdown[0].value for item in assistant_messages(app)] == ["First answer.", "Second answer.", "Third answer."]
    assert app.session_state["agent"] is agent
    last = network.bodies[-1]["contents"]
    assert last[0]["parts"][0]["text"] == "First"
    expected = ", ".join(row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"])
    assert f"the main set in the previous answer was {expected}." in last[-1]["parts"][0]["text"]


def test_new_chat_clears_the_conversation_and_keeps_the_connection(run_app):
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("First answer."),
        text_reply("Fresh answer."),
    )
    app = ask(connect(run_app(network)), "First")
    agent = app.session_state["agent"]
    requests_before = len(network.keys_seen)

    button(app, "New Chat").click().run()
    assert not app.exception
    assert not list(app.chat_message)
    assert sorted(card_questions(app)) == sorted(EXAMPLE_CARDS)
    assert app.session_state["history"] == []
    assert app.session_state["agent"] is agent
    assert agent.contents == [] and agent._evidence == [] and agent._main_set == []
    assert app.session_state["agent"].model == agent.model
    assert len(network.keys_seen) == requests_before

    ask(app, "Second")
    assert [item["role"] for item in network.bodies[-1]["contents"]] == ["user"]
    assert network.bodies[-1]["contents"][0]["parts"][0]["text"] == "Second"
    assert network.keys_seen[-1] == FAKE_KEY
    assert len(network.keys_seen) == requests_before + 1
    assert [item.markdown[0].value for item in assistant_messages(app)] == ["Fresh answer."]


def test_new_chat_clears_an_error(run_app):
    network = FakeNetwork({"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}, status=503)
    app = ask(connect(run_app(network)), "Question")
    assert app.error
    assert [item.markdown[0].value for item in user_messages(app)] == ["Question"]
    assert card_questions(app) == []
    button(app, "New Chat").click().run()
    assert not app.error
    assert not list(app.chat_message)
    assert sorted(card_questions(app)) == sorted(EXAMPLE_CARDS)


def test_score_ui_stays_in_the_message_that_reports_the_score(run_app):
    network = FakeNetwork(
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply("ANC is 2.6% long-haul."),
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("ILG leads."),
    )
    app = ask(ask(connect(run_app(network)), "Long haul?"), "Top scores?")
    long_haul, score = assistant_messages(app)
    assert not long_haul.info and not score_expanders(long_haul)
    assert len(score.info) == 1 and len(score_expanders(score)) == 1
    assert not score_expanders(score)[0].proto.expanded


def test_evidence_is_collapsed_inside_each_assistant_message(run_app):
    network = FakeNetwork(
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply("One."),
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}), text_reply("Two."),
    )
    app = ask(ask(connect(run_app(network)), "One?"), "Two?")
    for message in assistant_messages(app):
        evidence = evidence_expanders(message)
        assert len(evidence) == 1
        assert not evidence[0].proto.expanded
    assert len(evidence_expanders(app)) == 2
    assert not any(evidence_expanders(message) for message in user_messages(app))


def test_main_message_has_no_duplicate_tables(run_app):
    network = FakeNetwork(
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}), text_reply("SNA had more delays."),
    )
    app = ask(connect(run_app(network)), "Compare")
    assert len(main_frames(assistant_messages(app)[0])) == 1
    assert [item.value for item in direct(assistant_messages(app)[0], Markdown)].count("**Operational congestion**") == 1


def test_unmet_demand_stays_indirect_in_instructions_and_data():
    assert "Never give an amount for the unmeasured thing" in gemini_agent.SYSTEM_PROMPT
    assert "say in your first sentence that you cannot give it" in gemini_agent.SYSTEM_PROMPT
    assert "unmet demand" in gemini_agent.SYSTEM_PROMPT
    metrics = tools.get_airport_profile("SFO")["profile"]["demand_pressure_proxies"]["metrics"]
    assert not [name for name in metrics if "unmet" in name]


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
    assert not evidence[0].proto.expanded
    assert [tab.label for tab in app.tabs[:4]] == ["Results", "Definitions", "Sources", "Caveats"]
    assert not app.warning or "do not match" not in page_text(app)
    congestion = next(frame.value for frame in app.dataframe if "Average taxi-out (min)" in set(frame.value["Measure"]))
    row = congestion[congestion["Measure"] == "Departures delayed 15+ min (%)"].iloc[0]
    assert (row["LAX"], row["SNA"]) == (18.1, 19.2)
    assert "On-Time Performance" in page_text(app)
    assert "PROVISIONAL" in page_text(app)


def test_all_four_tools_render(run_app):
    network = FakeNetwork(
        function_call("rank_airports", {"window_years": 3, "states": ["MA"], "top_n": 3, "min_enplanements": 100000}),
        text_reply("Ranking."),
        function_call("get_airport_profile", {"airport": "SFO", "table_sections": ["congestion"]}),
        text_reply("Profile."),
        function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}),
        text_reply("Long haul."),
    )
    app = connect(run_app(network))
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
    assert first["National rank"] == "#1"
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
    assert table.iloc[0]["National rank"] == "#17"
    assert table.iloc[0][SCORE_LABEL] == 62.39
    assert table.iloc[0][presentation.POSITION_LABEL] == 1
    text = page_text(app)
    assert "Over 2,000,000 2025 departing passengers" in text
    assert "Filters do not change scores or national ranks" in text


def test_default_preset_adds_no_filter_to_the_question(run_app):
    network = RecordingNetwork(text_reply("Answer."))
    ask(connect(run_app(network)), "Which airports should we investigate for terminal expansion?")
    assert "Selected filter" not in json.dumps(network.bodies[0]["contents"])


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
    assert table["National rank"] == "#82"
    assert table["Absolute Passenger Growth 3Y"] == "5,924,745"
    assert table["Passenger CAGR 3Y (%)"] == "4.16"


def test_airport_without_a_score_says_why(run_app):
    network = FakeNetwork(function_call("get_airport_profile", {"airport": "JAC", "table_sections": ["score"]}), text_reply("JAC has no score."))
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


def test_answers_appear_in_order_with_evidence_collapsed(run_app):
    network = FakeNetwork(
        function_call("get_airport_profile", {"airport": "SFO"}), text_reply("First answer."),
        function_call("get_airport_profile", {"airport": "BOS"}), text_reply("Second answer."),
    )
    app = ask(ask(connect(run_app(network)), "First"), "Second")
    text = page_text(app)
    assert text.index("First answer.") < text.index("Second answer.")
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
    assert [len(score_expanders(message)) for message in assistant_messages(app)] == [0, 1]


def test_follow_up_with_an_unrelated_rank_shows_no_score_ui(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 5}), text_reply("ILG leads."),
        function_call("rank_airports", RANK_CAGR), text_reply("ORD is #3 by passenger CAGR."),
    )
    app = ask(ask(connect(run_app(network)), "Scores?"), "Growth?")
    assert [len(score_expanders(message)) for message in assistant_messages(app)] == [1, 0]


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
    first, second = two_scored_airports()
    network = FakeNetwork(profile_call(first, ["score"]), profile_call(second, ["score"]), text_reply("Neither is clear-cut."))
    app = ask(connect(run_app(network)), "Which first?")
    assert_no_score_ui(app)
    assert [item.value for item in app.caption].count(presentation.SCORE_PROXY_NOTE) == 2


def test_command_line_tables_keep_the_score_caption():
    result = tools.get_airport_profile("ATL", ["score"])
    call = ToolCall("get_airport_profile", {"airport": "ATL", "table_sections": ["score"]}, result, False)
    assert presentation.main_tables([call])[0].note == presentation.SCORE_PROXY_NOTE


SHORTLIST_TURN = [
    function_call("rank_opportunity_score", {"states": [], "top_n": 5}),
    function_call("rank_airports", {**RANK_CAGR, "top_n": 5}),
    text_reply("Start with this shortlist. Another airport is a supporting example from the growth ranking."),
]


def ranking_frames(block):
    return [frame.value for frame in block.dataframe if "Airport" in frame.value.columns]


def test_follow_up_keeps_the_exact_shortlist_and_stays_compact(run_app):
    shortlist = [row["airport"] for row in tools.rank_opportunity_score([], 5)["ranked"]]
    network = RecordingNetwork(
        *SHORTLIST_TURN,
        function_call("compare_airports", {"airports": shortlist, "table_sections": ["passenger_growth"]}),
        text_reply("ILM added the most passengers."),
    )
    app = ask(ask(connect(run_app(network)), "Where would you start?"), "Which of those grew the most?")

    # What the model received names the shown set, not the supporting example.
    note = network.bodies[3]["contents"][-1]["parts"][0]["text"]
    assert f"the main set in the previous answer was {', '.join(shortlist)}." in note
    growth_rows = tools.rank_airports(3, [], 5, 100000)["ranked"]
    supporting = {row["airport"] for row in growth_rows} - set(shortlist)
    assert supporting and not any(code in note for code in supporting)
    assert app.session_state["history"][1][0] == "Which of those grew the most?"

    follow_up = assistant_messages(app)[1]
    frames = main_frames(follow_up)
    assert len(frames) == 1
    table = frames[0].set_index("Measure")
    assert list(table.columns) == shortlist
    assert len(table) == 6
    assert "Total Load Factor 2025 (%), passengers / seats" not in table.index
    assert not score_expanders(follow_up)

    # The evidence stays on the same topic, and the complete result is kept for traceability.
    measures = {item for frame in evidence_expanders(follow_up)[0].dataframe for item in frame.value.get("Measure", [])}
    assert "Absolute Passenger Growth 3Y" in measures
    assert "Total Load Factor 2025 (%), passengers / seats" not in measures
    result = app.session_state["history"][1][1].tool_calls[0].result
    assert result["airports"][shortlist[0]]["demand_pressure_proxies"]["metrics"]["total_load_factor_2025_pct"] is not None
    assert len(evidence_expanders(app)) == 2


def test_screening_answer_shows_one_main_ranking_and_keeps_the_other_in_evidence(run_app):
    network = FakeNetwork(*SHORTLIST_TURN)
    app = ask(connect(run_app(network)), "Where would you start?")
    assistant = app.chat_message[1]
    frames = main_frames(assistant)
    assert len(frames) == 1
    assert SCORE_LABEL in frames[0].columns
    assert not any("Passenger CAGR 3Y (%)" in frame.columns and SCORE_LABEL not in frame.columns for frame in frames)
    assert "**Ranked by Passenger CAGR 3Y**" not in [item.value for item in direct(assistant, Markdown)]
    assert "**Ranked by Passenger CAGR 3Y**" in [item.value for item in app.markdown]
    assert_score_ui_once(app)


NEW_ENGLAND = ["CT", "MA", "ME", "NH", "RI", "VT"]


@pytest.mark.parametrize("score_first", [True, False], ids=["score_first", "growth_first"])
def test_score_table_leads_and_defines_the_follow_up_set_in_either_call_order(run_app, score_first):
    score = function_call("rank_opportunity_score", {"states": NEW_ENGLAND, "top_n": 5})
    growth = function_call("rank_airports", {**RANK_CAGR, "states": NEW_ENGLAND, "top_n": 10})
    first, second = (score, growth) if score_first else (growth, score)
    network = RecordingNetwork(
        first, second, text_reply("These rank highest for further investigation."),
        text_reply("Which group do you mean?"),
    )
    app = ask(ask(connect(run_app(network)), "Which airports look most promising?"), "Which of those grew most?")

    expected = [row["airport"] for row in tools.rank_opportunity_score(NEW_ENGLAND, 5)["ranked"]]
    answer = assistant_messages(app)[0]
    frames = main_frames(answer)
    assert len(frames) == 1
    assert SCORE_LABEL in frames[0].columns
    assert frames[0]["Airport"].tolist() == expected
    assert "**Ranked by Passenger CAGR 3Y**" in [item.value for item in app.markdown]
    assert "**Ranked by Passenger CAGR 3Y**" not in [item.value for item in direct(answer, Markdown)]
    assert len(answer.info) == 1 and len(score_expanders(answer)) == 1

    note = network.bodies[3]["contents"][-1]["parts"][0]["text"]
    assert f"the main set in the previous answer was {', '.join(expected)}." in note
    growth_only = {row["airport"] for row in tools.rank_airports(3, NEW_ENGLAND, 10, 100000)["ranked"]} - set(expected)
    assert growth_only and not any(code in note for code in growth_only)


def test_supporting_comparison_does_not_change_the_shortlist(run_app):
    ranked = [row["airport"] for row in tools.rank_opportunity_score(NEW_ENGLAND, 25)["ranked"]]
    expected, outside = ranked[:3], ranked[-2:]
    assert not set(expected) & set(outside)
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": NEW_ENGLAND, "top_n": 3}),
        function_call("compare_airports", {"airports": outside, "table_sections": ["congestion"]}),
        text_reply("Two other airports are supporting examples outside this list."),
        text_reply("Done."),
    )
    app = ask(ask(connect(run_app(network)), "Most promising?"), "Which of those?")
    note = network.bodies[3]["contents"][-1]["parts"][0]["text"]
    assert f"was {', '.join(expected)}." in note
    assert not any(code in note for code in outside)


@pytest.mark.parametrize("score_first", [True, False], ids=["score_first", "growth_first"])
def test_growth_answer_can_lead_with_growth_even_when_the_score_tool_ran(run_app, score_first):
    score = function_call("rank_opportunity_score", {"states": NEW_ENGLAND, "top_n": 5})
    growth = function_call("rank_airports", {**RANK_CAGR, "states": NEW_ENGLAND})
    first, second = (score, growth) if score_first else (growth, score)
    network = RecordingNetwork(
        first, second, text_reply("Growth was fastest at the top of this list.\n\n[primary_table: rank_airports]"),
        text_reply("Done."),
    )
    app = ask(ask(connect(run_app(network)), "Which grew fastest?"), "Which of those?")
    answer = assistant_messages(app)[0]
    frames = main_frames(answer)
    assert len(frames) == 1
    assert "Passenger CAGR 3Y (%)" in frames[0].columns and SCORE_LABEL not in frames[0].columns
    assert not answer.info and not score_expanders(answer)
    assert not any("primary_table" in str(item.value) for item in app.markdown)

    expected = [row["airport"] for row in tools.rank_airports(3, NEW_ENGLAND, 3, 100000)["ranked"]]
    assert frames[0]["Airport"].tolist() == expected
    assert f"was {', '.join(expected)}." in network.bodies[3]["contents"][-1]["parts"][0]["text"]

    # The unused score leaves the visible evidence, but its tool call and full result remain.
    assert "**Airport Opportunity Score**" not in [item.value for item in app.markdown]
    score_call = next(call for call in app.session_state["history"][0][1].tool_calls if call.name == "rank_opportunity_score")
    assert score_call.result["ranked"]
    assert "rank_opportunity_score" in "\n".join(item.value for item in app.markdown)


@pytest.mark.parametrize(
    "line", ["", "[primary_table: not_a_tool]", "[primary_table: rank_opportunity_score]"], ids=["absent", "invalid", "score"],
)
def test_score_leads_unless_the_model_validly_names_growth(run_app, line):
    network = FakeNetwork(
        function_call("rank_airports", {**RANK_CAGR, "states": NEW_ENGLAND}),
        function_call("rank_opportunity_score", {"states": NEW_ENGLAND, "top_n": 5}),
        text_reply(f"Top-ranked for further investigation.\n\n{line}"),
    )
    app = ask(connect(run_app(network)), "Which look most promising?")
    answer = assistant_messages(app)[0]
    assert SCORE_LABEL in main_frames(answer)[0].columns
    assert len(answer.info) == 1
    assert not any("primary_table" in str(item.value) for item in app.markdown)


def test_growth_led_answer_that_quotes_a_score_still_shows_the_disclaimer(run_app):
    row = tools.rank_opportunity_score(NEW_ENGLAND, 1)["ranked"][0]
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": NEW_ENGLAND, "top_n": 1}),
        function_call("rank_airports", {**RANK_CAGR, "states": NEW_ENGLAND}),
        text_reply(f"Growth leads here, and the top airport scored {row['opportunity_score']}.\n\n[primary_table: rank_airports]"),
    )
    app = ask(connect(run_app(network)), "Which grew fastest?")
    answer = assistant_messages(app)[0]
    assert SCORE_LABEL not in main_frames(answer)[0].columns
    assert len(answer.info) == 1 and len(score_expanders(answer)) == 1


def two_scored_airports():
    first, second = tools.rank_opportunity_score([], 2)["ranked"]
    return first["airport"], second["airport"]


def profile_call(code, sections):
    return function_call("get_airport_profile", {"airport": code, "table_sections": sections})


def assert_only_airport_in_main_table(message, shown, hidden):
    frames = main_frames(message)
    assert len(frames) == 1
    assert list(frames[0].columns) == ["Measure", shown]
    assert hidden not in frames[0].columns


@pytest.mark.parametrize("featured_is_last", [True, False], ids=["featured_last", "featured_first"])
def test_look_ups_show_one_table_for_the_airport_the_answer_features(run_app, featured_is_last):
    first, second = two_scored_airports()
    featured, other = (second, first) if featured_is_last else (first, second)
    network = RecordingNetwork(
        profile_call(first, ["score"]),
        profile_call(second, ["score", "demand"]),
        text_reply(f"I would look at {featured} first, then {other}.\n\nIts growth stands out."),
    )
    app = ask(connect(run_app(network)), "Which would you investigate first?")
    answer = assistant_messages(app)[0]
    assert_only_airport_in_main_table(answer, featured, other)
    assert network.generate_calls == 3
    # Nothing is lost: both look-ups and the demand table stay in the collapsed evidence.
    evidence_columns = [frame.value.columns for frame in evidence_expanders(answer)[0].dataframe]
    assert any(first in columns for columns in evidence_columns)
    assert any(second in columns for columns in evidence_columns)
    measures = [set(frame.value.get("Measure", [])) for frame in evidence_expanders(answer)[0].dataframe]
    assert any("Total Load Factor 2025 (%), passengers / seats" in item for item in measures)


@pytest.mark.parametrize("featured_is_last", [True, False], ids=["featured_last", "featured_first"])
def test_named_airport_wins_when_the_answer_mentions_another_first(run_app, featured_is_last):
    first, second = two_scored_airports()
    featured, other = (second, first) if featured_is_last else (first, second)
    network = FakeNetwork(
        profile_call(first, ["score"]), profile_call(second, ["score", "demand"]),
        text_reply(f"{other} has grown fastest, but I'd look at {featured}.\n\n[primary_table: {featured}]"),
    )
    app = ask(connect(run_app(network)), "Which would you investigate first?")
    answer = assistant_messages(app)[0]
    assert_only_airport_in_main_table(answer, featured, other)
    assert not any("primary_table" in str(item.value) for item in app.markdown)


def test_airport_code_in_the_line_can_be_lower_case(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(
        profile_call(first, ["score"]), profile_call(second, ["score"]),
        text_reply(f"{first} is fast, but {second} is my choice.\n\n[primary_table: {second.lower()}]"),
    )
    app = ask(connect(run_app(network)), "Which first?")
    assert_only_airport_in_main_table(assistant_messages(app)[0], second, first)


def test_named_airport_that_was_not_looked_up_shows_no_look_up_table(run_app):
    first, second, third = [row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"]]
    network = FakeNetwork(
        profile_call(first, ["score"]), profile_call(second, ["score"]),
        text_reply(f"I'd stay with {third}, based on what we already have.\n\n[primary_table: {third}]"),
    )
    app = ask(connect(run_app(network)), "Which first?")
    assert main_frames(assistant_messages(app)[0]) == []
    assert len(evidence_expanders(assistant_messages(app)[0])[0].dataframe) >= 2


def test_single_look_up_contrasted_with_an_earlier_airport_follows_the_line(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(
        profile_call(first, ["score"]),
        text_reply(f"{first} looks less convincing, so I'd keep {second}.\n\n[primary_table: {second}]"),
    )
    app = ask(connect(run_app(network)), "Which first?")
    assert main_frames(assistant_messages(app)[0]) == []


def test_comparison_with_a_named_airport_keeps_one_table_with_all_airports(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(
        function_call("compare_airports", {"airports": [first, second], "table_sections": ["score", "demand", "congestion"]}),
        text_reply(f"{first} and {second} differ, and I'd start with {second}.\n\n[primary_table: {second}]"),
    )
    app = ask(connect(run_app(network)), "Which first?")
    frames = main_frames(assistant_messages(app)[0])
    assert len(frames) == 1
    assert {first, second} <= set(frames[0].columns)


def test_screening_table_ignores_a_line_naming_one_airport(run_app):
    ranked = [row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"]]
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        profile_call(ranked[1], ["score"]),
        text_reply(f"I'd look at this shortlist.\n\n[primary_table: {ranked[1]}]"),
    )
    app = ask(connect(run_app(network)), "Which look most promising?")
    frames = main_frames(assistant_messages(app)[0])
    assert len(frames) == 1 and frames[0]["Airport"].tolist() == ranked


def test_look_up_is_matched_by_the_city_in_its_tool_result(run_app):
    first, second = two_scored_airports()
    city = tools.get_airport_profile(second)["profile"]["city"]
    network = FakeNetwork(
        profile_call(first, ["score"]), profile_call(second, ["score"]),
        text_reply(f"I would start with {city}.\n\nGrowth stands out."),
    )
    app = ask(connect(run_app(network)), "Which first?")
    assert_only_airport_in_main_table(assistant_messages(app)[0], second, first)


def test_look_ups_the_answer_does_not_name_show_no_main_table(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(profile_call(first, ["score"]), profile_call(second, ["score"]), text_reply("Neither is clear-cut."))
    app = ask(connect(run_app(network)), "Which first?")
    answer = assistant_messages(app)[0]
    assert main_frames(answer) == []
    assert_no_score_ui(app)
    assert len(evidence_expanders(answer)[0].dataframe) >= 2


def test_featured_airport_without_a_requested_table_gets_none(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(profile_call(first, ["score"]), profile_call(second, []), text_reply(f"I would start with {second}."))
    app = ask(connect(run_app(network)), "Which first?")
    assert main_frames(assistant_messages(app)[0]) == []


def test_repeated_look_up_of_one_airport_shows_one_table(run_app):
    first, _ = two_scored_airports()
    network = FakeNetwork(
        profile_call(first, ["score"]), profile_call(first, ["congestion"]), text_reply(f"{first} is the one to review."),
    )
    app = ask(connect(run_app(network)), "Which first?")
    assert len(main_frames(assistant_messages(app)[0])) == 1


def test_one_airport_answer_shows_one_compact_table_and_prefers_the_shorter(run_app):
    first, _ = two_scored_airports()
    network = FakeNetwork(profile_call(first, ["demand", "score", "congestion"]), text_reply(f"{first} grew quickly."))
    app = ask(connect(run_app(network)), "Tell me about it")
    answer = assistant_messages(app)[0]
    assert len(main_frames(answer)) == 1
    assert SCORE_LABEL in set(main_frames(answer)[0]["Measure"])


def test_one_airport_answer_can_still_show_the_demand_table_alone(run_app):
    first, _ = two_scored_airports()
    network = FakeNetwork(profile_call(first, ["demand"]), text_reply(f"{first} grew quickly."))
    app = ask(connect(run_app(network)), "Tell me about demand")
    assert "Total Load Factor 2025 (%), passengers / seats" in set(main_frames(assistant_messages(app)[0])[0]["Measure"])


def test_hidden_look_up_score_shows_no_score_ui_until_the_answer_states_it(run_app):
    first, second = two_scored_airports()
    score = tools.get_airport_profile(first)["profile"]["opportunity_score"]["opportunity_score"]
    quiet = FakeNetwork(profile_call(first, ["score"]), profile_call(second, []), text_reply(f"{second} has the clearer growth story."))
    assert_no_score_ui(ask(connect(run_app(quiet)), "Which first?"))
    stated = FakeNetwork(profile_call(first, ["score"]), profile_call(second, []), text_reply(f"{second} leads, though {first} scored {score}."))
    assert len(ask(connect(run_app(stated)), "Which first?").info) == 1


def test_screening_ranking_stays_full_when_the_answer_names_one_airport_and_looks_it_up(run_app):
    ranked = [row["airport"] for row in tools.rank_opportunity_score([], 5)["ranked"]]
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 5}),
        profile_call(ranked[0], ["score", "demand"]),
        text_reply(f"{ranked[0]} ranks highest for further investigation."),
    )
    app = ask(connect(run_app(network)), "Which look most promising?")
    frames = main_frames(assistant_messages(app)[0])
    assert len(frames) == 1
    assert frames[0]["Airport"].tolist() == ranked


def test_comparison_focused_on_one_airport_keeps_one_table(run_app):
    first, second = two_scored_airports()
    sections = ["score", "demand", "congestion"]
    focused = FakeNetwork(
        function_call("compare_airports", {"airports": [first, second], "table_sections": sections}),
        text_reply(f"{first} is where I would start."),
    )
    assert len(main_frames(assistant_messages(ask(connect(run_app(focused)), "Which first?"))[0])) == 1
    both = FakeNetwork(
        function_call("compare_airports", {"airports": [first, second], "table_sections": sections}),
        text_reply(f"{first} and {second} differ in several ways."),
    )
    assert len(main_frames(assistant_messages(ask(connect(run_app(both)), "Compare them"))[0])) == 3


def test_look_ups_inside_the_shortlist_keep_it_for_the_next_follow_up(run_app):
    ranked = [row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"]]
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        profile_call(ranked[2], []), profile_call(ranked[1], []), text_reply(f"I would start with {ranked[1]}."),
        text_reply("Done."),
    )
    app = connect(run_app(network))
    ask(ask(ask(app, "Start?"), "Which first?"), "What about the others?")
    last = network.bodies[-1]["contents"][-1]["parts"][0]["text"]
    assert f"the main set in the previous answer was {', '.join(ranked)}." in last


def test_featured_airport_alone_starts_the_new_set_when_look_ups_leave_the_shortlist(run_app):
    rows = [row["airport"] for row in tools.rank_opportunity_score([], 5)["ranked"]]
    shortlist, outside = rows[:2], rows[2:4]
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 2}), text_reply("Shortlist."),
        profile_call(outside[0], []), profile_call(outside[1], []),
        text_reply(f"I'd go with {outside[1]}.\n\n[primary_table: {outside[1]}]"),
        text_reply("Done."),
    )
    app = connect(run_app(network))
    ask(ask(ask(app, "Start?"), "What about others?"), "And its growth?")
    assert f"the main set in the previous answer was {outside[1]}." in network.bodies[-1]["contents"][-1]["parts"][0]["text"]


def test_look_up_outside_the_shortlist_starts_a_new_set(run_app):
    ranked = [row["airport"] for row in tools.rank_opportunity_score([], 3)["ranked"]]
    outside = tools.rank_opportunity_score([], 4)["ranked"][3]["airport"]
    network = RecordingNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("Shortlist."),
        profile_call(outside, []), text_reply(f"{outside} is different."),
        text_reply("Done."),
    )
    app = ask(ask(ask(connect(run_app(network)), "Start?"), f"What about {outside}?"), "And its growth?")
    assert f"the main set in the previous answer was {outside}." in network.bodies[-1]["contents"][-1]["parts"][0]["text"]


def test_answer_first_reply_with_rounded_tool_numbers_renders_cleanly(run_app):
    rows = tools.rank_opportunity_score([], 3)["ranked"]
    cities = [tools.get_airport_profile(row["airport"])["profile"]["city"] for row in rows]
    lead = rows[0]
    reply = (
        f"I'd start with {cities[0]}, {cities[1]} and {cities[2]}, but for different reasons.\n\n"
        f"{cities[0]} has added around {round(lead['absolute_passenger_growth_3y'] / 1000)} thousand departing passengers, "
        f"with traffic growing about {round(lead['passenger_cagr_3y_pct'])}% a year.\n\n"
        "These are leads to investigate, not proof that more capacity is needed."
    )
    network = FakeNetwork(function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply(reply))
    app = ask(connect(run_app(network)), "Where would you look?")
    answer = assistant_messages(app)[0]
    children = list(answer.children.values())
    assert children[0].value.startswith(f"I'd start with {cities[0]}")
    assert isinstance(children[1], Markdown) and children[1].value == f"**{presentation.SCORE_TABLE_TITLE}**"
    assert main_frames(answer)[0]["Airport"].tolist() == [row["airport"] for row in rows]
    assert app.session_state["history"][0][1].unverified_numbers == []
    assert not any("do not match any tool result" in item.value for item in app.warning)
    assert len(answer.info) == 1


def evidence_titles(expander):
    return [item.value.strip("*") for item in expander.markdown if item.value.startswith("**") and item.value.endswith("**")]


def evidence_text(expander):
    parts = [item.value for group in (expander.markdown, expander.caption, expander.warning) for item in group]
    return "\n".join(parts)


def test_congestion_comparison_shows_only_congestion_evidence(run_app):
    first, second = two_scored_airports()
    network = FakeNetwork(
        function_call("compare_airports", {"airports": [first, second], "table_sections": ["congestion"]}),
        text_reply(f"{first} and {second} differ on delays."),
    )
    app = ask(connect(run_app(network)), "Compare congestion")
    expander = evidence_expanders(assistant_messages(app)[0])[0]
    assert evidence_titles(expander) == ["Operational congestion"]
    text = evidence_text(expander)
    # Its definitions, coverage and sources are present.
    assert "Average taxi-out (min)" in text
    assert "shows the data's coverage" in text
    assert tools.SOURCES["ontime"] in text
    assert tools.CONGESTION_SCOPE in text
    # Nothing from other topics is shown in the tables, definitions, sources or caveats.
    for unrelated in (
        tools.SOURCES["faa"], tools.SOURCES["t100_routes"], tools.CARGO_INCLUSIVE, tools.SIZE_BIAS, tools.LOAD_FACTOR,
        "Passenger CAGR 3Y (%)", "Airport Opportunity Score", "points by metric", "Cargo context", "Freight",
    ):
        assert unrelated not in text
    assert not any(SCORE_LABEL in set(frame.value.get("Measure", [])) for frame in expander.dataframe)


def test_raw_output_is_kept_for_verification_but_not_displayed(run_app):
    first, second = two_scored_airports()
    score = tools.get_airport_profile(first)["profile"]["opportunity_score"]["opportunity_score"]
    network = FakeNetwork(
        function_call("compare_airports", {"airports": [first, second], "table_sections": ["congestion"]}),
        text_reply(f"{first} scored {score} and {second} has 1,234,567 invented passengers."),
    )
    app = ask(connect(run_app(network)), "Compare")
    assert "Raw" not in [tab.label for tab in app.tabs]
    assert not list(app.get("json"))
    answer = app.session_state["history"][0][1]
    # The unrequested score still verifies, and the full result stays with the answer.
    assert answer.unverified_numbers == ["1,234,567"]
    assert answer.tool_calls[0].result["airports"][first]["opportunity_score"]["opportunity_score"] == score
    assert presentation.build_evidence(answer.tool_calls[0]).raw is answer.tool_calls[0].result
    # The tool call itself remains visible for traceability.
    assert "tool `compare_airports`" in "\n".join(item.value for item in app.markdown)


def test_focused_evidence_keeps_the_complete_tool_result():
    first, second = [row["airport"] for row in tools.rank_opportunity_score([], 2)["ranked"]]
    result = tools.compare_airports([first, second], ["congestion"])
    call = ToolCall("compare_airports", {}, result, False)
    full = presentation.build_evidence(call)
    focused = presentation.focused_evidence(full, call, [call], "No numbers here.")
    assert [table.title for table in focused.tables] == ["Operational congestion"]
    assert focused.raw is result
    assert result["airports"][first]["opportunity_score"]["opportunity_score"] is not None
    assert len(full.tables) > len(focused.tables)


def test_screening_evidence_keeps_the_score_and_supporting_metrics(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 3}),
        function_call("rank_airports", {**RANK_CAGR, "top_n": 3}),
        text_reply("These rank highest for further investigation."),
    )
    app = ask(connect(run_app(network)), "Which look most promising?")
    titles = evidence_titles(evidence_expanders(assistant_messages(app)[0])[0])
    assert presentation.SCORE_TABLE_TITLE in titles
    assert "Score points by metric" in titles
    assert "Ranked by Passenger CAGR 3Y" in titles


def test_look_up_with_no_table_shows_only_the_section_the_answer_cites(run_app):
    first, _ = two_scored_airports()
    profile = tools.get_airport_profile(first)["profile"]
    congestion = next(
        code for code in (row["airport"] for row in tools.rank_opportunity_score([], 25)["ranked"])
        if tools.get_airport_profile(code)["profile"]["operational_congestion"]
    )
    taxi = tools.get_airport_profile(congestion)["profile"]["operational_congestion"]["average_taxi_out_min"]
    network = FakeNetwork(profile_call(congestion, []), text_reply(f"Taxi-out at {congestion} averages {taxi} minutes."))
    app = ask(connect(run_app(network)), "How long is taxi-out?")
    assert evidence_titles(evidence_expanders(assistant_messages(app)[0])[0]) == ["Operational congestion"]


def test_growth_figure_brings_in_the_compact_growth_table_not_the_score(run_app):
    first, _ = two_scored_airports()
    cagr = tools.get_airport_profile(first)["profile"]["demand_pressure_proxies"]["metrics"]["passenger_cagr_3y_pct"]
    network = FakeNetwork(profile_call(first, []), text_reply(f"{first} grew {cagr}% a year."))
    app = ask(connect(run_app(network)), "How fast?")
    titles = evidence_titles(evidence_expanders(assistant_messages(app)[0])[0])
    assert titles[0] == "Passenger growth"
    assert presentation.SCORE_TABLE_TITLE not in titles
    assert presentation.DEMAND_TABLE_TITLE not in titles


def test_quoted_score_brings_in_the_score_evidence(run_app):
    first, _ = two_scored_airports()
    score = tools.get_airport_profile(first)["profile"]["opportunity_score"]["opportunity_score"]
    network = FakeNetwork(profile_call(first, []), text_reply(f"{first} scored {score}."))
    app = ask(connect(run_app(network)), "Its score?")
    assert presentation.SCORE_TABLE_TITLE in evidence_titles(evidence_expanders(assistant_messages(app)[0])[0])


def test_look_up_with_no_cited_numbers_explains_where_the_data_is(run_app):
    first, _ = two_scored_airports()
    network = FakeNetwork(profile_call(first, []), text_reply(f"{first} looks worth a closer look."))
    app = ask(connect(run_app(network)), "Worth a look?")
    expander = evidence_expanders(assistant_messages(app)[0])[0]
    assert evidence_titles(expander) == []
    assert "This answer did not rely on a table from this tool call." in evidence_text(expander)
    assert "tool `get_airport_profile`" in evidence_text(expander)


def test_small_base_notice_survives_focused_evidence(run_app):
    code = next(row["airport"] for row in tools.rank_opportunity_score([], 25)["ranked"] if row["small_base_warnings"])
    network = FakeNetwork(profile_call(code, []), text_reply(f"{code} grew quickly."))
    app = ask(connect(run_app(network)), "Why so fast?")
    assert [item.value for item in app.warning if item.value.startswith(f"Small base year: {code} (")]


def long_haul_answer(run_app, reply):
    network = FakeNetwork(function_call("get_long_haul_share", {"airport": "ANC", "threshold_miles": 3000}), text_reply(reply))
    app = ask(connect(run_app(network)), "What is the percentage of long haul flights out of Anchorage airport?")
    return app, assistant_messages(app)[0]


def test_long_haul_answer_shows_each_flight_group_in_a_visible_table(run_app):
    app, answer = long_haul_answer(run_app, "It depends on which flights you mean.")
    frames = main_frames(answer)
    assert len(frames) == 1
    rows = frames[0].set_index("Flights")
    assert rows.index.tolist() == ["All flights in file", "Passenger aircraft", "Freighters"]
    assert rows["Long-haul share (%)"].tolist() == [23.0, 2.7, 47.7]
    assert rows["Departures"].tolist() == [43066, 23474, 19109]
    assert rows["Long-haul departures"].tolist() == [9900, 623, 9123]


def test_long_haul_table_never_labels_the_all_flight_share_as_passenger(run_app):
    _, answer = long_haul_answer(run_app, "It depends on which flights you mean.")
    rows = main_frames(answer)[0].set_index("Flights")
    assert rows.loc["Passenger aircraft", "Long-haul share (%)"] != 23.0
    assert rows.loc["All flights in file", "Long-haul share (%)"] == 23.0


def test_long_haul_caption_states_threshold_period_scope_and_range(run_app):
    _, answer = long_haul_answer(run_app, "It depends on which flights you mean.")
    captions = [item.value for item in direct(answer, Caption)]
    assert len(captions) == 1
    caption = captions[0]
    assert "at least 3,000 statute miles" in caption
    assert "January\u2013July 2026" in caption
    assert "US carriers only" in caption
    assert "could be 19.8% to 33.6%" in caption
    assert "86.3% of ANC's departures" in caption


def test_long_haul_evidence_keeps_the_sensitivity_and_coverage_detail(run_app):
    _, answer = long_haul_answer(run_app, "It depends on which flights you mean.")
    titles = evidence_titles(evidence_expanders(answer)[0])
    assert titles == ["ANC long-haul share", "Route file coverage"]
    assert any("Share at 1500+ (%)" in frame.value.columns for frame in evidence_expanders(answer)[0].dataframe)


def test_long_haul_table_says_so_when_coverage_is_unavailable():
    result = tools.get_long_haul_share("ANC", 3000)
    result["coverage"] = {"available": False, "reason": "Summary does not cover all route-file months"}
    note = presentation.main_tables([ToolCall("get_long_haul_share", {}, result, False)])[0].note
    assert "could not be calculated" in note
    assert "could be" not in note


def test_long_haul_table_uses_the_requested_threshold():
    result = tools.get_long_haul_share("ANC", 1500)
    table = presentation.main_tables([ToolCall("get_long_haul_share", {}, result, False)])[0]
    assert "at least 1,500 statute miles" in table.note
    assert table.data.set_index("Flights").loc["All flights in file", "Long-haul share (%)"] == 37.9


@pytest.mark.parametrize(
    "period, label",
    [
        ({"first": "2026-01", "last": "2026-07"}, "January\u2013July 2026"),
        ({"first": "2026-03", "last": "2026-03"}, "March 2026"),
        ({"first": "2025-11", "last": "2026-02"}, "November 2025 to February 2026"),
    ],
)
def test_period_labels(period, label):
    assert presentation._period_label(period) == label


def test_congestion_definitions_describe_the_actual_denominators(run_app):
    network = FakeNetwork(
        function_call("compare_airports", {"airports": ["LAX", "SNA"], "table_sections": ["congestion"]}), text_reply("LAX and SNA differ."),
    )
    app = ask(connect(run_app(network)), "Compare")
    text = "\n".join(item.value for item in evidence_expanders(assistant_messages(app)[0])[0].markdown)
    assert "NAS delay (min per scheduled arrival)" in text
    for measure in ("nas", "weather", "carrier", "late_aircraft"):
        definition = presentation.CONGESTION_DEFINITIONS[f"{measure}_delay_min_per_arrival"]
        assert definition.endswith("/ scheduled arrivals, including cancelled flights")
        assert definition in text
    assert "including cancelled flights" in presentation.CONGESTION_DEFINITIONS["departures_in_sample"]
    assert "can exceed 100%" in presentation.CONGESTION_DEFINITIONS["share_of_t100_domestic_departures_pct"]
    assert "Departure counts include cancelled flights" in tools.CONGESTION_SCOPE


def test_documentation_states_the_verification_limits():
    # Markdown wraps lines, so compare the words rather than the layout.
    sources = " ".join((config.ROOT / "docs" / "DATA_SOURCES.md").read_text().split())
    assert "35 of 516 FAA codes" in sources
    assert "144 hub airports" in sources
    assert "have no history in this snapshot" in sources and "IWA and GPI, which are small hubs with full history" in sources
    assert "no raw copy or hash was kept" in sources
    assert "The cause has not been" in sources and "KTN" in sources
    assert "142 hub airports" not in sources and "36 of 516" not in sources


def test_growth_question_keeps_the_growth_table_without_score_ui(run_app):
    network = FakeNetwork(
        function_call("rank_airports", {**RANK_CAGR, "states": NEW_ENGLAND}), text_reply("HVN grew fastest."),
    )
    app = ask(connect(run_app(network)), "Which New England airports grew fastest?")
    answer = assistant_messages(app)[0]
    assert "Passenger CAGR 3Y (%)" in main_frames(answer)[0].columns
    assert SCORE_LABEL not in main_frames(answer)[0].columns
    assert_no_score_ui(app)


def test_visible_ranks_are_short_and_evidence_keeps_the_peer_group(run_app):
    network = FakeNetwork(function_call("rank_opportunity_score", {"states": [], "top_n": 3}), text_reply("ILG leads."))
    app = ask(connect(run_app(network)), "Top?")
    assert main_frames(app.chat_message[1])[0]["National rank"].tolist() == ["#1", "#2", "#3"]
    all_ranks = [rank for frame in ranking_frames(app) if "National rank" in frame.columns for rank in frame["National rank"]]
    assert all_ranks.count("#1 of 229") == 1


def test_short_ranks_leave_other_values_and_the_input_unchanged():
    frame = pd.DataFrame({"National rank": ["#1 of 229"], "Reason": ["2 of 12 months"], "Passengers": [100]})
    table = presentation.Table("Ranks", frame)
    short = presentation.with_short_ranks([table])[0].data.iloc[0]
    assert (short["National rank"], short["Reason"], short["Passengers"]) == ("#1", "2 of 12 months", 100)
    assert frame.iloc[0]["National rank"] == "#1 of 229"


def test_passenger_growth_table_copies_tool_values_and_stays_out_of_defaults():
    result = tools.compare_airports(["LAX", "SNA"], ["passenger_growth"])
    call = ToolCall("compare_airports", {}, result, False)
    tables = presentation.main_tables([call])
    assert [table.title for table in tables] == ["Passenger growth"]
    row = tables[0].data.set_index("Measure")["LAX"]
    metrics = result["airports"]["LAX"]["demand_pressure_proxies"]["metrics"]
    assert row["Absolute Passenger Growth 3Y"] == f"{metrics['absolute_passenger_growth_3y']:,}"
    assert row["Passenger CAGR 3Y (%)"] == f"{metrics['passenger_cagr_3y_pct']:.2f}"

    default = tools.compare_airports(["LAX", "SNA"])
    titles = [table.title for table in presentation.main_tables([ToolCall("compare_airports", {}, default, False)])]
    assert "Passenger growth" not in titles


def test_passenger_growth_table_reports_a_missing_year_without_estimating():
    result = tools.get_airport_profile("JAC", ["passenger_growth"])
    row = presentation.main_tables([ToolCall("get_airport_profile", {}, result, False)])[0].data.set_index("Measure")["JAC"]
    assert row["Passenger CAGR 3Y (%)"] == "n/a"
    assert row["Absolute Passenger Growth 3Y"] == "n/a"


def test_each_answer_decides_for_itself(run_app):
    network = FakeNetwork(
        function_call("rank_opportunity_score", {"states": [], "top_n": 1}), text_reply("ILG leads."),
        function_call("rank_airports", RANK_CAGR), text_reply("Fastest growth."),
        function_call("rank_opportunity_score", {"states": ["CA"], "top_n": 1}), text_reply("SFO leads."),
    )
    app = connect(run_app(network))
    ask(ask(ask(app, "First"), "Second"), "Third")
    assert_score_ui_once(app, answers=2)
    assert [len(score_expanders(message)) for message in assistant_messages(app)] == [1, 0, 1]


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
    assert "temporarily overloaded or unavailable" in app.error[0].value and "try again" in app.error[0].value.lower()
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
