"""Gemini conversation loop. Same tools and safeguards as the Anthropic agent."""

from __future__ import annotations

import copy
import json

import httpx
from google import genai
from google.genai import types

from . import tools
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

# Stable Flash models from Google's model list, newest first.
PREFERRED_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
MAX_TOOL_ROUNDS = 3
# The SDK waits forever without this, so a stalled request would never end.
REQUEST_TIMEOUT_MS = 60_000
NO_RESPONSE_MESSAGE = (
    "Gemini did not respond in time. Nothing was retried automatically. "
    "You can try again or try another available model."
)


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
    try:
        return {
            model.name.removeprefix("models/")
            for model in client.models.list()
            if "generateContent" in (model.supported_actions or [])
        }
    except httpx.TransportError as error:
        raise AgentError(NO_RESPONSE_MESSAGE) from error


def _no_preferred_model(available: set[str]) -> AgentError:
    flash = sorted(name for name in available if "flash" in name)
    return AgentError(f"No preferred Flash model is available. Available Flash models: {flash}. Set GEMINI_MODEL.")


def usable_models(client: genai.Client, extra: str | None = None) -> list[str]:
    """List the preferred models this key can use, newest first, plus one named model if it is available."""
    available = _available_models(client)
    models = [model for model in dict.fromkeys((*PREFERRED_MODELS, extra)) if model in available]
    if not models:
        raise _no_preferred_model(available)
    return models


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

    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model
        self.contents: list[types.Content] = []
        self._evidence: list[str] = []
        self._main_set: list[str] = []
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

    def reset(self) -> None:
        """Start a new conversation with the same client."""
        self.contents = []
        self._evidence = []
        self._main_set = []

    def ask(self, question: str) -> Answer:
        turn_start = len(self.contents)
        evidence_start = len(self._evidence)
        self.contents.append(
            types.Content(role="user", parts=[types.Part(text=with_main_set_note(question, self._main_set))])
        )
        self._evidence.append(question)
        try:
            answer = self._run_turn()
        except Exception:
            # Remove the failed turn so the next question starts from valid history.
            del self.contents[turn_start:]
            del self._evidence[evidence_start:]
            raise
        self._main_set = next_main_set(self._main_set, answer.tool_calls, answer.primary_table)
        return answer

    def _generate(self, config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        try:
            return self.client.models.generate_content(model=self.model, contents=self.contents, config=config)
        except httpx.TransportError as error:
            raise AgentError(NO_RESPONSE_MESSAGE) from error

    def _run_turn(self) -> Answer:
        calls: list[ToolCall] = []
        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            final_round = round_number == MAX_TOOL_ROUNDS
            response = self._generate(self._final_config if final_round else self._config)
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
                return answer

            results = []
            for call in function_calls:
                arguments = _clean_args(call.args)
                result, is_error = tools.execute_tool(call.name, arguments)
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
