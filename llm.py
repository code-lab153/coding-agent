"""Minimal synchronous model communication through the Responses API."""

from dataclasses import dataclass
import json
from typing import Any

from openai import OpenAI

from config import ModelConfig, load_model_config


Message = dict[str, str]
_ALLOWED_ROLES = {"system", "user", "assistant"}


class LLMError(RuntimeError):
    """Raised for invalid messages or provider-response failures."""


@dataclass(frozen=True)
class ToolCall:
    """Provider-independent description of one requested function call."""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    """Provider-independent text and requested function calls."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient:
    """Send ordinary text messages through the synchronous OpenAI client."""

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        """Load configuration and initialize the provider SDK client."""
        self._config = model_config or load_model_config()
        client_options: dict[str, str] = {"api_key": self._config.api_key}
        if self._config.base_url is not None:
            client_options["base_url"] = self._config.base_url

        try:
            self._client = OpenAI(**client_options)
        except Exception:
            raise LLMError("Could not initialize the model API client.") from None

    def send(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    ) -> LLMResponse:
        """Send messages and normalize text plus requested function calls."""
        normalized_messages = _validate_messages(messages)
        request_options: dict[str, Any] = {
            "model": self._config.model_name,
            "input": normalized_messages,
        }
        if tools:
            request_options["tools"] = list(tools)
            request_options["parallel_tool_calls"] = False

        try:
            provider_response: Any = self._client.responses.create(**request_options)
        except Exception:
            raise LLMError(
                "Model request failed. Check the network, endpoint, credentials, "
                "and model access."
            ) from None

        raw_text = getattr(provider_response, "output_text", None)
        text = raw_text if isinstance(raw_text, str) and raw_text.strip() else ""
        tool_calls = _parse_tool_calls(provider_response)
        if not text and not tool_calls:
            raise LLMError("Model response did not contain text or tool calls.")
        return LLMResponse(text=text, tool_calls=tool_calls)


def _validate_messages(messages: list[Message]) -> list[Message]:
    """Validate and copy the small message format accepted in Phase 3."""
    if not isinstance(messages, list) or not messages:
        raise LLMError("Messages must be a non-empty list.")

    normalized: list[Message] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMError(f"Message {index} must be a dictionary.")

        role = message.get("role")
        content = message.get("content")
        if role not in _ALLOWED_ROLES:
            raise LLMError(f"Message {index} has an unsupported role.")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Message {index} must contain non-empty text.")
        normalized.append({"role": role, "content": content})
    return normalized


def _parse_tool_calls(provider_response: object) -> tuple[ToolCall, ...]:
    """Normalize provider function-call output items without executing them."""
    output_items = getattr(provider_response, "output", None)
    if output_items is None:
        return ()
    if not isinstance(output_items, (list, tuple)):
        raise LLMError("Model response output items were malformed.")

    tool_calls: list[ToolCall] = []
    for item in output_items:
        if _item_value(item, "type") != "function_call":
            continue

        call_id = _item_value(item, "call_id")
        name = _item_value(item, "name")
        arguments_json = _item_value(item, "arguments")
        if not isinstance(call_id, str) or not call_id.strip():
            raise LLMError("Function call is missing a valid call_id.")
        if not isinstance(name, str) or not name.strip():
            raise LLMError("Function call is missing a valid name.")
        if not isinstance(arguments_json, str):
            raise LLMError("Function call arguments must be JSON text.")

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            raise LLMError(f"Function call {call_id} has invalid JSON arguments.") from None
        if not isinstance(arguments, dict):
            raise LLMError(f"Function call {call_id} arguments must be a JSON object.")

        tool_calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
    return tuple(tool_calls)


def _item_value(item: object, field: str) -> object:
    """Read a field from either an SDK object or a small test dictionary."""
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)
