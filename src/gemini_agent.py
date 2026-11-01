"""Gemini conversation loop. Same tools and safeguards as the Anthropic agent."""

from __future__ import annotations

import copy
import json
import logging
import re
import sys
import time

import httpx
from google import genai
from google.genai import errors as gemini_errors
from google.genai import types

from . import grounding, tools
from .agent import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    Answer,
    AgentError,
    ToolCall,
    build_answer,
    next_main_set,
    with_main_set_note,
)

# Stable Flash models from Google's model list, in priority order.
PREFERRED_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash")
MAX_TOOL_ROUNDS = 3
# The SDK waits forever without this, so a stalled request would never end.
REQUEST_TIMEOUT_MS = 60_000
NO_RESPONSE_MESSAGE = "Gemini did not respond in time. Nothing was retried automatically. You can try again."
MODELS_UNAVAILABLE_MESSAGE = "Gemini isn't available right now. Please try again in a little while."
TIMING_LOGGER = logging.getLogger("airport.timing")


def enable_timing_log() -> None:
    """Write timing lines to the server's stderr. Safe to call on every page run."""
    TIMING_LOGGER.setLevel(logging.INFO)
    if not TIMING_LOGGER.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s TIMING %(message)s"))
        TIMING_LOGGER.addHandler(handler)


def log_timing(event: str, **fields: object) -> None:
    """Log one event. Callers pass only durations, counts, status codes, model IDs and tool names."""
    TIMING_LOGGER.info("%s %s", event, " ".join(f"{name}={value}" for name, value in fields.items()))


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def build_client(api_key: str) -> genai.Client:
    """Create a client with a timeout and retries off, so one call never becomes several."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _available_models(client: genai.Client) -> set[str]:
    """List the models the key can generate with. Listing models generates no content."""
    started = time.perf_counter()
    status = "ok"
    try:
        return {
            model.name.removeprefix("models/")
            for model in client.models.list()
            if "generateContent" in (model.supported_actions or [])
        }
    except httpx.TransportError as error:
        status = "no_response"
        raise AgentError(NO_RESPONSE_MESSAGE) from error
    except gemini_errors.APIError as error:
        status = error.code
        raise
    finally:
        log_timing("gemini_list_models", status=status, duration_ms=elapsed_ms(started))


def _no_preferred_model(available: set[str]) -> AgentError:
    flash = sorted(name for name in available if "flash" in name)
    return AgentError(f"No preferred Flash model is available. Available Flash models: {flash}. Set GEMINI_MODEL.")


def usable_models(client: genai.Client, extra: str | None = None) -> list[str]:
    """List the preferred models this key can use, in priority order, plus one named model if it is available."""
    available = _available_models(client)
    models = [model for model in dict.fromkeys((*PREFERRED_MODELS, extra)) if model in available]
    if not models:
        raise AgentError(MODELS_UNAVAILABLE_MESSAGE)
    return models


def model_pair(client: genai.Client, requested: str | None = None) -> tuple[str, str | None]:
    """Pick the model to start with and at most one fallback. A named model that is available goes first."""
    models = usable_models(client, requested)
    if requested in models:
        models = [requested, *(model for model in models if model != requested)]
    return models[0], models[1] if len(models) > 1 else None


def quota_scope(error: gemini_errors.APIError) -> str | None:
    """Read whether Google's quota details name a per-model limit. None means they do not say.

    A limit is per-model only if every violation is, because any other would survive a model change.
    """
    body = error.details.get("error", error.details) if isinstance(error.details, dict) else {}
    entries = body.get("details") if isinstance(body, dict) else None
    violations = [
        violation
        for entry in entries or []
        if isinstance(entry, dict)
        for violation in entry.get("violations") or []
        if isinstance(violation, dict) and ("quotaId" in violation or "quotaDimensions" in violation)
    ]
    if not violations:
        return None
    per_model = all(
        "model" in (violation.get("quotaDimensions") or {}) or "PerModel" in str(violation.get("quotaId"))
        for violation in violations
    )
    return "model" if per_model else "key"


def is_model_specific(error: gemini_errors.APIError) -> bool:
    """Tell whether another model could clearly avoid this error. Anything unclear counts as no."""
    if error.code == 404:
        return True
    if error.code == 429:
        return quota_scope(error) == "model"
    return error.code == 503 and bool(re.search(r"\bmodels?\b", error.message or "", re.IGNORECASE))


def select_model(client: genai.Client, requested: str | None = None) -> str:
    """Pick a model the key can actually use."""
    available = _available_models(client)
    if requested:
        if requested not in available:
            raise AgentError(f"Model {requested!r} is not available to this API key.")
        return requested
    for model in PREFERRED_MODELS:
        if model in available:
            return model
    raise _no_preferred_model(available)


def _gemini_schema(input_schema: dict) -> dict:
    """Adapt a tool schema without changing the Python tool it describes.

    Allowed integer values move into the description because Gemini enums are
    documented for strings. The Python tools still validate every input.
    """
    schema = copy.deepcopy(input_schema)
    schema.pop("additionalProperties", None)
    for prop in schema["properties"].values():
        if prop.get("type") == "integer" and "enum" in prop:
            allowed = ", ".join(str(value) for value in prop.pop("enum"))
            prop["description"] = f"{prop['description']}. Allowed values: {allowed}"
    return schema


def tool_declarations() -> list[types.Tool]:
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=_gemini_schema(tool["input_schema"]),
            )
            for tool in tools.TOOL_DEFINITIONS
        ])
    ]


def _clean_args(args: dict | None) -> dict:
    """Convert JSON whole-number floats such as 3.0 to integers."""
    return {
        key: int(value) if isinstance(value, float) and value.is_integer() else value
        for key, value in (args or {}).items()
    }


class GeminiAgent:
    """Hold one conversation so follow-up questions keep their context."""

    def __init__(self, client: genai.Client, model: str, fallback_model: str | None = None):
        self.client = client
        self.model = model
        self._models = (model, fallback_model)
        self._fallback_model = fallback_model
        self.contents: list[types.Content] = []
        self._evidence: list[str] = []
        self._main_set: list[str] = []
        self._retrieved: set[str] = set()
        self._turn = self._new_turn_timing()
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
            tools=tool_declarations(),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
        )
        # The last call of a turn must answer from the tool results already returned.
        self._final_config = self._config.model_copy(update={
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
            ),
        })
        # Used once, when an answer lacks data. The model must call a tool but still chooses which.
        self._retrieval_config = self._config.model_copy(update={
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
            ),
        })

    def reset(self) -> None:
        """Start a new conversation with the same client, back on the preferred model."""
        self.model, self._fallback_model = self._models
        self.contents = []
        self._evidence = []
        self._main_set = []
        self._retrieved = set()

    @staticmethod
    def _new_turn_timing() -> dict:
        return {"requests": 0, "request_ms": 0, "tools": 0, "tool_ms": 0, "fallback": "no", "grounding_retry": "no"}

    def _log_question(self, started: float, outcome: str) -> None:
        total_ms = elapsed_ms(started)
        turn = self._turn
        log_timing(
            "question_total",
            duration_ms=total_ms,
            **turn,
            other_ms=max(0, total_ms - turn["request_ms"] - turn["tool_ms"]),
            outcome=outcome,
        )

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        self._turn = self._new_turn_timing()
        turn_start = len(self.contents)
        evidence_start = len(self._evidence)
        self.contents.append(
            types.Content(role="user", parts=[types.Part(text=with_main_set_note(question, self._main_set))])
        )
        self._evidence.append(question)
        try:
            answer = self._run_turn(question)
        except Exception as error:
            # Remove the failed turn so the next question starts from valid history.
            del self.contents[turn_start:]
            del self._evidence[evidence_start:]
            self._log_question(started, type(error).__name__)
            raise
        self._main_set = next_main_set(self._main_set, answer.tool_calls, answer.primary_table)
        self._retrieved |= grounding.retrieved_airports(answer.tool_calls)
        self._log_question(started, "answered")
        return answer

    def _request(self, config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        started = time.perf_counter()
        status = "ok"
        usage = None
        try:
            response = self.client.models.generate_content(model=self.model, contents=self.contents, config=config)
            usage = response.usage_metadata
            return response
        except httpx.TransportError as error:
            status = "no_response"
            raise AgentError(NO_RESPONSE_MESSAGE) from error
        except gemini_errors.APIError as error:
            status = error.code
            raise
        finally:
            duration = elapsed_ms(started)
            self._turn["requests"] += 1
            self._turn["request_ms"] += duration
            log_timing(
                "gemini_request",
                model=self.model,
                status=status,
                duration_ms=duration,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                thinking_tokens=getattr(usage, "thoughts_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )

    def _generate(self, config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        """Send one request, moving to the fallback model at most once per conversation.

        Only the failed request is resent, so no tool runs again. The history goes back unchanged,
        because Google manages thought signatures across models when their blocks are resent.
        """
        try:
            return self._request(config)
        except gemini_errors.APIError as error:
            if self._fallback_model is None or not is_model_specific(error):
                raise
            log_timing("gemini_fallback", from_model=self.model, to_model=self._fallback_model, status=error.code)
            self._turn["fallback"] = "yes"
        self.model, self._fallback_model = self._fallback_model, None
        try:
            return self._request(config)
        except gemini_errors.APIError as error:
            if is_model_specific(error):
                raise AgentError(MODELS_UNAVAILABLE_MESSAGE) from error
            raise

    def _run_turn(self, question: str) -> Answer:
        calls: list[ToolCall] = []
        last_round = MAX_TOOL_ROUNDS
        retried = False
        round_number = 0
        while round_number < last_round:
            round_number += 1
            final_round = round_number == last_round
            if final_round:
                config = self._final_config
            elif retried:
                config = self._retrieval_config
            else:
                config = self._config
            response = self._generate(config)
            if not response.candidates or response.candidates[0].content is None:
                feedback = response.prompt_feedback
                reason = feedback.block_reason if feedback else "no content returned"
                raise AgentError(f"Gemini returned no answer ({reason}). Try rephrasing the question.")

            candidate = response.candidates[0]
            # The content goes back unchanged, which preserves thought signatures.
            self.contents.append(candidate.content)
            parts = candidate.content.parts or []
            function_calls = [part.function_call for part in parts if part.function_call]

            if function_calls and final_round:
                # Mode NONE should prevent this. Run nothing; ask() removes the failed turn.
                break

            if not function_calls:
                text = "\n".join(part.text for part in parts if part.text and not part.thought).strip()
                answer = build_answer(text, calls, self._evidence)
                if not answer.text:
                    raise AgentError(f"Gemini returned no text (finish reason: {candidate.finish_reason}).")
                missing = grounding.unsupported(question, answer.text, calls, self._retrieved, self._evidence)
                if missing is None:
                    return answer
                if retried or final_round:
                    # ask() removes the failed turn, so the unsupported draft leaves no trace.
                    raise AgentError(grounding.withheld_message(missing))
                # Drop the draft so it cannot anchor the retry, then require retrieval and one final answer.
                self.contents.pop()
                retried = True
                self._turn["grounding_retry"] = "yes"
                log_timing("grounding_retry", round=round_number)
                last_round = round_number + 2
                continue

            results = []
            for call in function_calls:
                arguments = _clean_args(call.args)
                started = time.perf_counter()
                result, is_error = tools.execute_tool(call.name, arguments)
                duration = elapsed_ms(started)
                self._turn["tools"] += 1
                self._turn["tool_ms"] += duration
                log_timing("tool", name=call.name, is_error=is_error, duration_ms=duration)
                content = tools.to_json(result)
                calls.append(ToolCall(call.name, arguments, result, is_error))
                self._evidence.append(content)
                results.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response=json.loads(content),
                )))
            self.contents.append(types.Content(role="user", parts=results))
        raise AgentError(f"No answer after {MAX_TOOL_ROUNDS} tool rounds. Try a narrower question.")
