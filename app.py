"""Streamlit interface. Run with: streamlit run app.py"""

from __future__ import annotations

import json
import os
import time

import streamlit as st
from google.genai import errors as gemini_errors
from streamlit.errors import StreamlitSecretNotFoundError

from src import cli, gemini_agent, presentation, tools
from src.agent import AgentError, Answer

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]
MISSING_KEY_MESSAGE = (
    "Gemini is not configured yet. Set the `GEMINI_API_KEY` environment variable, or add it to "
    "`.streamlit/secrets.toml`, then reload this page."
)
# Buttons have no height option, so this keeps the example cards the same size.
EXAMPLE_CARD_STYLE = """<style>
.st-key-example-cards button { min-height: 4.5rem; }
.st-key-example-cards button p { text-align: left; }
</style>"""

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

gemini_agent.enable_timing_log()

state = st.session_state
state.setdefault("agent", None)
state.setdefault("history", [])
state.setdefault("error", None)
state.setdefault("connection_error", None)
state.setdefault("failed_question", None)


def server_api_key() -> str:
    """Read the key from the server's environment or Streamlit secrets. It never goes to the browser."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def use_example(question: str) -> None:
    """Send an example through the same path as a typed question."""
    state.pending_question = question


def new_chat() -> None:
    """Forget the conversation but keep the connection."""
    state.agent.reset()
    state.history = []
    state.error = None
    state.failed_question = None


def retry_connection() -> None:
    state.connection_error = None


def connect(key: str) -> None:
    """List the models this key can use, which generates no content, and choose them automatically."""
    try:
        client = gemini_agent.build_client(key)
        model, fallback_model = gemini_agent.model_pair(client, os.environ.get("GEMINI_MODEL"))
    except AgentError as error:
        state.connection_error = str(error)
    except gemini_errors.APIError as error:
        state.connection_error = f"Could not connect to Gemini. {cli._gemini_error(error)}"
    else:
        state.agent = gemini_agent.GeminiAgent(client, model, fallback_model)


def ask(question: str) -> None:
    """Make one attempt, which the agent may resend once to its fallback model. A failure keeps the question to resend."""
    state.failed_question = question
    try:
        answer = state.agent.ask(question)
    except AgentError as error:
        state.error = str(error)
    except gemini_errors.APIError as error:
        state.error = cli._gemini_error(error)
    else:
        state.history.append((question, answer))
        state.error = None
        state.failed_question = None


def show_tool_evidence(evidence: presentation.ToolEvidence, key: str) -> None:
    st.markdown(f"**{evidence.label}** · tool `{evidence.tool}`")
    st.code(json.dumps(evidence.inputs), language="json")
    if evidence.error:
        st.warning(f"The tool rejected this input: {evidence.error}")
        return
    if evidence.periods:
        st.caption("Period: " + "; ".join(evidence.periods))
    for warning in evidence.warnings:
        st.warning(warning)

    results, definitions, sources, caveats = st.tabs(["Results", "Definitions", "Sources", "Caveats"])
    with results:
        if not evidence.tables:
            st.caption("This answer did not rely on a table from this tool call.")
        for table in evidence.tables:
            st.markdown(f"**{table.title}**")
            st.dataframe(table.data, hide_index=True, width="stretch")
            if table.note:
                st.caption(table.note)
    with definitions:
        for name, definition in evidence.definitions.items():
            label = presentation.LABELS.get(name, name.replace("_", " ").capitalize())
            st.markdown(f"**{label}:** {presentation.escape_markdown(definition)}")
    with sources:
        for source in evidence.sources:
            st.markdown(f"- {source}")
    with caveats:
        for caveat in evidence.provisional:
            st.warning(caveat)
        for caveat in evidence.caveats:
            st.markdown(f"- {presentation.escape_markdown(caveat)}")


def show_score_context() -> None:
    st.info("**The Airport Opportunity Score is a preliminary screening proxy, not an investment recommendation.**", icon="ℹ️")
    with st.expander("How the Airport Opportunity Score works"):
        st.markdown(tools.SCORE_DISCLAIMER)
        st.markdown(
            "The score combines 11 weighted metrics. Each airport's value is clipped to the 5th and 95th percentiles of a "
            "fixed national peer group, then scaled from 0 to 100. Scores are points out of 100, not probabilities. "
            "Filters never change a score or national rank."
        )
        st.dataframe(presentation.methodology_table(), hide_index=True, width="stretch")
        st.caption(tools.PEER_GROUP)
        st.markdown(f"{tools.SIZE_BIAS} {tools.BASE_2022} {tools.METRIC_OVERLAP}")


def show_question(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(presentation.escape_markdown(question))


def show_answer(answer: Answer, earlier_calls: list, key: str) -> None:
    analysis, closing = presentation.split_answer(answer.text)
    reports_score = presentation.reports_opportunity_score(
        answer.tool_calls, f"{analysis}\n\n{closing}", earlier_calls, answer.primary_table
    )

    def displayed(tables: list[presentation.Table]) -> list[presentation.Table]:
        return presentation.without_score_proxy_note(tables) if reports_score else tables

    with st.chat_message("assistant"):
        st.markdown(presentation.escape_markdown(presentation.hide_unverified_numbers(analysis, answer.unverified_numbers)))
        for table in presentation.with_contextual_headings(presentation.with_short_ranks(
            displayed(presentation.main_tables(answer.tool_calls, answer.primary_table, answer.text))
        )):
            st.markdown(f"**{table.title}**")
            st.dataframe(table.data, hide_index=True, width="stretch")
            if table.note:
                st.caption(table.note)
        if closing:
            st.markdown(presentation.escape_markdown(presentation.hide_unverified_numbers(closing, answer.unverified_numbers)))
        notice = presentation.small_base_notice(answer.tool_calls, answer.text)
        if notice:
            st.warning(notice, icon="⚠️")
        if not answer.tool_calls:
            st.warning("No tool was called, so no data result supports this answer.")
        if reports_score:
            show_score_context()
        if answer.tool_calls:
            with st.expander("Explore the data & sources"):
                for position, call in enumerate(answer.tool_calls):
                    evidence = presentation.focused_evidence(
                        presentation.build_evidence(call), call, answer.tool_calls, answer.text, answer.primary_table
                    )
                    evidence.tables = displayed(evidence.tables)
                    show_tool_evidence(evidence, f"{key}-{position}")


api_key = server_api_key()
if api_key and state.agent is None and state.connection_error is None:
    connect(api_key)

title_column, action_column = st.columns([5, 1], vertical_alignment="center")
with title_column:
    st.title("Airport Investment Intelligence")
    st.caption("Screening analysis of US airports using public BTS and FAA data.")
with action_column:
    st.button(
        "New Chat",
        on_click=new_chat,
        disabled=state.agent is None or not (state.history or state.error or state.failed_question),
        width="stretch",
    )

if not api_key:
    st.warning(MISSING_KEY_MESSAGE)
elif state.connection_error:
    st.error(state.connection_error)
    st.button("Try again", on_click=retry_connection)

question_input = st.chat_input("Ask about US airports", disabled=state.agent is None)
submitted = (state.pop("pending_question", None) or question_input or "").strip()

if not state.history and not submitted and not state.failed_question:
    st.subheader("What would you like to explore?")
    st.html(EXAMPLE_CARD_STYLE)
    with st.container(key="example-cards"):
        example_columns = st.columns(2)
        for position, example in enumerate(EXAMPLES):
            with example_columns[position % 2]:
                st.button(
                    example,
                    on_click=use_example,
                    args=(example,),
                    key=f"example-{position}",
                    disabled=state.agent is None,
                    width="stretch",
                )

render_started = time.perf_counter()
for position, (question, answer) in enumerate(state.history):
    earlier_calls = [call for _, earlier in state.history[:position] for call in earlier.tool_calls]
    show_question(question)
    show_answer(answer, earlier_calls, key=str(position))
if state.history:
    gemini_agent.log_timing("page_render", answers=len(state.history), duration_ms=gemini_agent.elapsed_ms(render_started))

if submitted:
    # Drawn once while waiting. The rerun then draws it from the history, so nothing appears twice.
    show_question(submitted)
    with st.chat_message("assistant"), st.spinner("Choosing a tool and analyzing the data..."):
        ask(submitted)
    st.rerun()

if state.failed_question:
    show_question(state.failed_question)
if state.error:
    st.error(state.error)
if state.failed_question and state.agent is not None:
    st.button("Ask again", on_click=use_example, args=(state.failed_question,))
