"""Gemini conversation loop. Same tools and safeguards as the Anthropic agent."""

from __future__ import annotations

import copy
import json
import os

import httpx
from google import genai
from google.genai import types

from . import tools
from .agent import MAX_TOKENS, SYSTEM_PROMPT, Answer, AgentError, ToolCall, find_unverified_numbers

# Stable Flash models from Google's model list, newest first.
PREFERRED_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
MAX_TOOL_ROUNDS = 3
DEFAULT_CALL_LIMIT = 6
# The SDK waits forever without this, so a stalled request would never end.
REQUEST_TIMEOUT_MS = 60_000
NO_RESPONSE_MESSAGE = (
    "Gemini did not respond in time. Nothing was retried automatically. "
    "You can try again or connect with another model."
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


def select_model(client: genai.Client, requested: str | None = None) -> str:
    """Pick a model the key can actually use. Listing models generates no content."""
    try:
        available = {
            model.name.removeprefix("models/")
            for model in client.models.list()
            if "generateContent" in (model.supported_actions or [])
        }
    except httpx.TransportError as error:
        raise AgentError(NO_RESPONSE_MESSAGE) from error
    if requested:
        if requested not in available:
            raise AgentError(f"Model {requested!r} is not available to this API key.")
        return requested
    for model in PREFERRED_MODELS:
        if model in available:
            return model
    flash = sorted(name for name in available if "flash" in name)
    raise AgentError(f"No preferred Flash model is available. Available Flash models: {flash}. Set GEMINI_MODEL.")


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

    def __init__(self, client: genai.Client, model: str, call_limit: int | None = None):
        self.client = client
        self.model = model
        self.call_limit = call_limit or int(os.environ.get("GEMINI_MAX_CALLS", DEFAULT_CALL_LIMIT))
        self.calls_used = 0
        self.contents: list[types.Content] = []
        self._evidence: list[str] = []
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
            tools=tool_declarations(),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
        )

    def ask(self, question: str) -> Answer:
        turn_start = len(self.contents)
        evidence_start = len(self._evidence)
        self.contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        self._evidence.append(question)
        try:
            return self._run_turn()
        except Exception:
            # Remove the failed turn so the next question starts from valid history.
            del self.contents[turn_start:]
            del self._evidence[evidence_start:]
            raise

    def _generate(self) -> types.GenerateContentResponse:
        if self.calls_used >= self.call_limit:
            raise AgentError(
                f"Stopped at the session limit of {self.call_limit} Gemini calls. "
                "Set GEMINI_MAX_CALLS to change it."
            )
        # Count before calling, because a failed request may still use allowance.
        self.calls_used += 1
        try:
            return self.client.models.generate_content(model=self.model, contents=self.contents, config=self._config)
        except httpx.TransportError as error:
            raise AgentError(NO_RESPONSE_MESSAGE) from error

    def _run_turn(self) -> Answer:
        calls: list[ToolCall] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._generate()
            if not response.candidates or response.candidates[0].content is None:
                feedback = response.prompt_feedback
                reason = feedback.block_reason if feedback else "no content returned"
                raise AgentError(f"Gemini returned no answer ({reason}). Try rephrasing the question.")

            candidate = response.candidates[0]
            # The content goes back unchanged, which preserves thought signatures.
            self.contents.append(candidate.content)
            parts = candidate.content.parts or []
            function_calls = [part.function_call for part in parts if part.function_call]

            if not function_calls:
                text = "\n".join(part.text for part in parts if part.text and not part.thought).strip()
                if not text:
                    raise AgentError(f"Gemini returned no text (finish reason: {candidate.finish_reason}).")
                return Answer(
                    text=text,
                    tool_calls=calls,
                    unverified_numbers=find_unverified_numbers(text, self._evidence),
                )

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
