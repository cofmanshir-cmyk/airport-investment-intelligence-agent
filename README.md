# Airport Investment Intelligence Agent

Ask airport investment questions in plain English. A real LLM chooses a deterministic Python
tool, the tool calculates the result on public BTS and FAA data, and the LLM explains it.

This is a working demo. The **Airport Opportunity Score** is a preliminary screening proxy, not
investment advice or a return forecast. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Getting started

Run the web app with your own Gemini API key. You need Python 3.11 or newer. The commands are for
macOS, Linux and Codespaces; on Windows, use `.venv\Scripts\` in place of `.venv/bin/`.

**1. Clone the repository and install the dependencies.**

```bash
git clone https://github.com/cofmanshir-cmyk/airport-investment-intelligence-agent.git
cd airport-investment-intelligence-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**2. Get a Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey). Questions use
your own key and count against its Gemini quota and any billing on your account.

**3. Create `.streamlit/secrets.toml`** in the project folder, using a text editor, with this one line.
Replace `your-key-here` with your key and keep the straight quotes:

```toml
GEMINI_API_KEY = "your-key-here"
```

This is a local file that git ignores, and it **must never be committed**. Don't force-add it or copy
the key into any other project file. If a key is ever committed, revoke it in Google AI Studio and
create a new one.

If the `GEMINI_API_KEY` environment variable is set, including as a Codespaces secret, it takes
precedence over this file. An old value there silently replaces the key in the file, so update or
unset it if the app uses the wrong key.

**4. Run the app** from the project folder:

```bash
.venv/bin/streamlit run app.py
```

Open <http://localhost:8501> in your browser. In Codespaces, open the **Ports** tab, then open port 8501,
and keep the port's visibility **Private**. The key stays on the server and never reaches the browser.

### How the web app uses Gemini

The page chooses the model itself: the first two of `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`,
`gemini-3.6-flash` and `gemini-3.8-flash` that the key can use become the preferred model and its one
fallback. To start with another available model, set `GEMINI_MODEL` on the server. If Gemini reports a
problem specific to the preferred model (not found, a per-model quota, or an overloaded model), the
failed request is resent once to the fallback, and the conversation stays on it. No tool runs again.
Key, account-wide quota and other errors are not retried. **New Chat** clears the conversation and
returns to the preferred model. The example cards on an empty chat send their question straight to the
agent.

The server writes `TIMING` lines to stderr with each Gemini request's duration, model and token counts,
each tool's duration, and whether a fallback or grounding retry occurred. They never include keys,
questions or answers.

Gemini can answer without calling a tool. If that answer names an airport or states a figure with no
retrieved data behind it, the agent discards it and makes one further attempt that requires a tool
call, which costs one or two extra Gemini calls. If there is still no data, no answer is shown. General
questions about definitions or methods stay free of tools. The check confirms data was retrieved for the
airport, not that every sentence is right, so it doesn't replace the number check or the evidence section.

## Optional: command-line app

The web app doesn't need this section. After step 1 above, you can also chat in the terminal, using
Anthropic or Gemini. The command-line app doesn't read `.streamlit/secrets.toml`.

### Use Anthropic

```bash
.venv/bin/python -m src.cli
```

You need an API key from the [Anthropic Console](https://console.anthropic.com/). The app asks
for it with hidden input and keeps it in memory only. To avoid typing it each time, set
`ANTHROPIC_API_KEY` as a [Codespaces secret](https://docs.github.com/en/codespaces/managing-your-codespace/managing-your-account-specific-secrets-for-github-codespaces).
Never put an Anthropic key in a repository file.

Each question calls the paid API, using `claude-opus-5-5` by default. Set `ANTHROPIC_MODEL` to
use another model.

### Use Gemini

```bash
.venv/bin/python -m src.cli --provider gemini
```

The app asks for your Gemini API key the same way, or uses the `GEMINI_API_KEY` environment variable,
confirms which Flash model the key can use, and then uses the same tools and safeguards. Automatic
retries of failed requests are off, and Google enforces its own quota; if it runs out, you see a clear
message and nothing is resent. Set `GEMINI_MODEL` to choose a model.

## Data

- **Demand:** BTS T-100, 2015, 2022 and 2025
- **Congestion:** BTS On-Time Performance, 2025
- **Long-haul:** T-100 route file, Jan-Jul 2026, US carriers only
- **Eligibility:** FAA CY2025 enplanements

All calculations run in Python. The LLM chooses tools and explains their results. See
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for details and limits. Each answer's **Explore the
data & sources** section shows the sources, definitions and caveats behind that answer.

## Example questions

- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
- Which airports have the highest Airport Opportunity Score?
- Which airports added the most passengers over 3 years?
- Which airports with over 2 million departing passengers score highest?
- Which airports have the fastest 3-year passenger growth?
- Rank New England airports by 10-year passenger growth.
- Compare LAX and SNA congestion.
- Give me a profile of SFO.
- What percentage of flights departing ANC are long-haul?

Follow-ups work, such as "what about at 2,400 miles?". On the command line, type `/raw` to see the
last tool results as JSON, or `/quit` to exit.

## How it works

| Layer | File | Role |
|---|---|---|
| Interfaces | `app.py`, `src/cli.py` | Web chat using Gemini, and command-line chat, each with evidence generated by code |
| Presentation | `src/presentation.py` | Builds tables and evidence from tool output; hides flagged numbers |
| LLM loops | `src/agent.py`, `src/gemini_agent.py` | The LLM selects tools and explains; unmatched numbers are flagged; Gemini model fallback |
| Grounding | `src/grounding.py` | Checks that a Gemini answer rests on data retrieved for the airports involved |
| Tools | `src/tools.py` | Validate inputs and attach definitions, sources and caveats |
| Calculations | `src/analytics.py` | Deterministic Python, with no LLM calls |
| Score | `src/scoring.py` | 11 weighted metrics against a fixed national peer group |
| Data | `src/pipeline.py` | Fetches, validates and snapshots public data |

Answers lead with a short conclusion. Results tables are built in Python from tool output, and the
LLM writes only the interpretation. Full evidence stays in a collapsed section.

The number check is a limited safeguard, not full verification. It flags some figures that match
no tool result and shows "[figure not verified]" in their place in the answer. The command line also
lists them in a warning. It can't tell whether a figure belongs to the airport or measure it's attached
to, whole numbers often match something by chance, and a sound rounding or calculation that matches no
tool result is also replaced. Compare important results with the evidence section, or with `/raw` on
the command line.

## Provisional and unavailable

- **Departure metrics include cargo flights.** They make up 35% of the score.
- **Absolute passenger growth favors larger airports.** Read it alongside Passenger CAGR.
- **2022 was a recovery year,** which may inflate 3-year growth.
- **Small base years are flagged, not excluded.** ILG and PAE are examples.
- **Eligibility:** the 100,000 enplanement threshold is provisional.
- **Incomplete years:** shown as null, and an airport missing a scored metric gets no score.
- **Service class:** passenger totals include all classes.
- **Not measured:** passenger-only capacity, terminal capacity, unmet demand, profitability and returns.
- **Source verification:** route, FAA and on-time data reconcile exactly with their raw files. The T-100
  summary, which supplies the score's inputs, was fetched from an API with no raw copy kept, so its 2015
  and 2022 values can't be checked against an original file.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the score and
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for sources and limitations.

## Tests

```bash
.venv/bin/python -m pytest
```

Tests use a scripted stand-in for the LLM, so they need no API key.

To rebuild the data, put the route file in `data/raw/` and run `.venv/bin/python -m src.pipeline`.
