"""Streamlit interface. Run with: streamlit run app.py"""

from __future__ import annotations

import json
import os

import streamlit as st
from google.genai import errors as gemini_errors

from src import cli, gemini_agent, presentation, tools
from src.agent import AgentError, Answer

AUTOMATIC_MODEL = "Automatic (newest available)"
ALL_AIRPORTS = "All eligible airports"
PASSENGER_PRESETS = {
    ALL_AIRPORTS: 0,
    "At least 500,000": 500_000,
    "At least 1 million": 1_000_000,
    "At least 5 million": 5_000_000,
    "At least 10 million": 10_000_000,
}

EXAMPLES = {
    "Airport Opportunity Score": "Which airports have the highest Airport Opportunity Score?",
    "Airport ranking": "Rank New England airports by 3-year passenger growth.",
    "Airport comparison": "Compare LAX and SNA congestion.",
    "Airport profile": "Give me a profile of SFO.",
    "Long-haul analysis": "What percentage of flights departing ANC are long-haul?",
}

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

state = st.session_state
state.setdefault("agent", None)
state.setdefault("history", [])
state.setdefault("error", None)


def use_example(question: str) -> None:
    """Fill in the question box. Nothing is sent until Ask is selected."""
    state.draft = question


def disconnect() -> None:
    state.agent = None
    state.history = []
    state.error = None


def connect(key: str, model_choice: str) -> None:
    """Check the key by listing models, which generates no content."""
    key = key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        state.error = "Enter a Gemini API key to connect."
        return
    requested = None if model_choice == AUTOMATIC_MODEL else model_choice
    try:
        client = gemini_agent.build_client(key)
        model = gemini_agent.select_model(client, requested)
    except AgentError as error:
        state.error = str(error)
    except gemini_errors.APIError as error:
        state.error = f"Could not connect. {cli._gemini_error(error)}"
    else:
        state.agent = gemini_agent.GeminiAgent(client, model)
        state.history = []
        state.error = None


def with_selected_filter(question: str, preset: str) -> str:
    """Pass a selected size filter to the model. The model still chooses the tools."""
    minimum = PASSENGER_PRESETS[preset]
    if not minimum:
        return question
    return f"{question}\n\n(Selected filter for rankings: at least {minimum:,} 2025 departing passengers.)"


def ask(question: str, preset: str = ALL_AIRPORTS) -> None:
    """Make one attempt. A failure is reported and is never retried automatically."""
    try:
        answer = state.agent.ask(with_selected_filter(question, preset))
    except AgentError as error:
        state.error = str(error)
    except gemini_errors.APIError as error:
        state.error = cli._gemini_error(error)
        if error.code == 503:
            state.error += " This is usually temporary. You can try again later or connect with another model."
    else:
        state.history.append((question, answer))
        state.error = None


def show_tool_evidence(evidence: presentation.ToolEvidence, key: str) -> None:
    st.markdown(f"**{evidence.label}** · tool `{evidence.tool}`")
    st.code(json.dumps(evidence.inputs), language="json")
    if evidence.error:
        st.warning(f"The tool rejected this input: {evidence.error}")
        return
    if evidence.disclaimer:
        st.warning(evidence.disclaimer)
    if evidence.periods:
        st.caption("Period: " + "; ".join(evidence.periods))
    for warning in evidence.warnings:
        st.warning(warning)

    results, definitions, sources, caveats, raw = st.tabs(["Results", "Definitions", "Sources", "Caveats", "Raw"])
    with results:
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
    with raw:
        st.json(evidence.raw, expanded=False)


def show_answer(question: str, answer: Answer, latest: bool, key: str) -> None:
    with st.chat_message("user"):
        st.markdown(presentation.escape_markdown(question))
    with st.chat_message("assistant"):
        conclusion, rest = presentation.split_answer(answer.text)
        st.markdown(presentation.escape_markdown(conclusion))
        for table in presentation.main_tables(answer.tool_calls):
            st.markdown(f"**{table.title}**")
            st.dataframe(table.data, hide_index=True, width="stretch")
            if table.note:
                st.caption(table.note)
        if rest:
            st.markdown(presentation.escape_markdown(rest))
        notice = presentation.small_base_notice(answer.tool_calls, answer.text)
        if notice:
            st.warning(notice, icon="⚠️")
        if answer.unverified_numbers:
            st.warning(
                "These numbers do not match any tool result, so verify them: " + ", ".join(answer.unverified_numbers)
            )
        if not answer.tool_calls:
            st.warning("No tool was called, so no data result supports this answer.")
    if answer.tool_calls:
        with st.expander("Evidence, methodology, sources and caveats (generated by Python, not the LLM)"):
            for position, call in enumerate(answer.tool_calls):
                show_tool_evidence(presentation.build_evidence(call), f"{key}-{position}")


with st.sidebar:
    st.header("Connection")
    if state.agent is None:
        models = [AUTOMATIC_MODEL, *gemini_agent.PREFERRED_MODELS]
        preferred = os.environ.get("GEMINI_MODEL")
        if preferred and preferred not in models:
            models.append(preferred)
        with st.form("connect", clear_on_submit=True):
            key_input = st.text_input(
                "Gemini API key",
                type="password",
                help="Kept in memory for this browser session only. It is never saved.",
            )
            model_choice = st.selectbox("Model", models, index=models.index(preferred) if preferred else 0)
            connect_clicked = st.form_submit_button("Connect", width="stretch")
        if os.environ.get("GEMINI_API_KEY"):
            st.caption("Leave the key blank to use GEMINI_API_KEY from this environment.")
        st.caption("Connecting checks which models your key can use. It generates no content.")
        if connect_clicked:
            connect(key_input, model_choice)
            st.rerun()
    else:
        agent = state.agent
        st.success(f"Connected to {agent.model}")
        st.metric("Gemini calls this session", f"{agent.calls_used} of {agent.call_limit}")
        st.caption(
            "A question usually takes 2 calls, and failed requests are counted. This counter covers this "
            "connection only, so keep track of your total allowance separately."
        )
        st.button("Disconnect", on_click=disconnect, width="stretch")

    st.divider()
    st.header("About the data")
    st.markdown(
        "- **Demand:** BTS T-100, 2015, 2022 and 2025\n"
        "- **Congestion:** BTS On-Time Performance, 2025\n"
        "- **Long-haul:** T-100 route file, Jan-Jul 2026, US carriers only\n"
        "- **Eligibility:** FAA CY2025 enplanements"
    )
    st.caption("All calculations run in Python. The LLM chooses tools and explains their results.")

st.title("Airport Investment Intelligence")
st.markdown("Screening analysis of US airports using public BTS and FAA data.")
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

st.markdown("**Example questions**")
st.caption("Selecting one fills in the question box. Nothing is sent until you select Ask.")
example_columns = st.columns(2)
for position, (label, question) in enumerate(EXAMPLES.items()):
    with example_columns[position % 2]:
        st.button(f"{label}: {question}", on_click=use_example, args=(question,), key=f"example-{position}", width="stretch")

with st.form("question", clear_on_submit=True):
    question_input = st.text_area("Your question", key="draft", height=80, placeholder="Ask about US airports")
    passenger_preset = st.selectbox(
        "Minimum 2025 departing passengers (optional)",
        list(PASSENGER_PRESETS),
        help="This limits which airports are shown. It never changes a score or national rank. "
        "You can also ask for any other size in your question.",
    )
    ask_clicked = st.form_submit_button("Ask", type="primary", disabled=state.agent is None)
if state.agent is None:
    st.caption("Connect in the sidebar to ask a question. On a narrow screen, open the sidebar with the arrow at the top left.")

if ask_clicked and question_input.strip():
    with st.spinner("Choosing a tool and analyzing the data..."):
        ask(question_input.strip(), passenger_preset)
    st.rerun()

if state.error:
    st.error(state.error)

if state.history:
    st.divider()
    for position in range(len(state.history) - 1, -1, -1):
        question, answer = state.history[position]
        show_answer(question, answer, latest=position == len(state.history) - 1, key=str(position))
        if position:
            st.divider()
