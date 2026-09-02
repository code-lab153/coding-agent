"""Synchronous model communication through the Responses API."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import json
from typing import Any

from openai import OpenAI

from config import ModelConfig, load_model_config


InputItem = dict[str, object]
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
    """Provider-independent model output needed by the Agent."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    continuation_items: tuple[InputItem, ...] = ()


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
        messages: list[InputItem],
        tools: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    ) -> LLMResponse:
        """Send messages and normalize text plus requested function calls."""
        #检查消息是不是合法
        normalized_messages = _validate_input_items(messages)
        #准备请求，包括模型，以及对应的消息
        request_options: dict[str, Any] = {
            "model": self._config.model_name,
            "input": normalized_messages,
        }
        #如果Agent有工具，需要加入工具和对应的参数
        if tools:
            request_options["tools"] = list(tools)
            request_options["parallel_tool_calls"] = False

        try:
            #实际上会传入当前整理好的上下文作为message，以及模型可以申请使用的工具说明，然后得到相应的返回结果，并包装成LLMResponse对象返回
            provider_response: Any = self._client.responses.create(**request_options)
        except Exception:
            raise LLMError(
                "Model request failed. Check the network, endpoint, credentials, "
                "and model access."
            ) from None

        raw_text = getattr(provider_response, "output_text", None)
        text = raw_text if isinstance(raw_text, str) and raw_text.strip() else ""
        tool_calls = _parse_tool_calls(provider_response)
        continuation_items = _serialize_output_items(provider_response)
        if not text and not tool_calls:
            raise LLMError("Model response did not contain text or tool calls.")
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            continuation_items=continuation_items,
        )


def _validate_input_items(messages: list[InputItem]) -> list[InputItem]:
    """Validate and copy plain message and continuation input items."""
    if not isinstance(messages, list) or not messages:
        raise LLMError("Messages must be a non-empty list.")

    normalized: list[InputItem] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMError(f"Message {index} must be a dictionary.")

        item_type = message.get("type")
        if item_type is None:
            role = message.get("role")
            content = message.get("content")
            if role not in _ALLOWED_ROLES:
                raise LLMError(f"Message {index} has an unsupported role.")
            if not isinstance(content, str) or not content.strip():
                raise LLMError(f"Message {index} must contain non-empty text.")
            normalized.append({"role": role, "content": content})
            continue

        if not isinstance(item_type, str) or not item_type.strip():
            raise LLMError(f"Message {index} has an invalid item type.")
        if item_type == "function_call_output":
            call_id = message.get("call_id")
            output = message.get("output")
            if not isinstance(call_id, str) or not call_id.strip():
                raise LLMError(f"Message {index} has an invalid call_id.")
            if not isinstance(output, str):
                raise LLMError(f"Message {index} must contain string tool output.")

        plain_item = _to_plain_data(message)
        if not isinstance(plain_item, dict):
            raise LLMError(f"Message {index} could not be serialized.")
        normalized.append(plain_item)
    return normalized


def _serialize_output_items(provider_response: object) -> tuple[InputItem, ...]:
    """Convert all provider output items into replayable plain dictionaries."""
    output_items = getattr(provider_response, "output", None)
    if output_items is None:
        return ()
    if not isinstance(output_items, (list, tuple)):
        raise LLMError("Model response output items were malformed.")

    continuation_items: list[InputItem] = []
    for item in output_items:
        plain_item = _to_plain_data(item)
        if not isinstance(plain_item, dict):
            raise LLMError("Model response contained an unserializable output item.")
        continuation_items.append(plain_item)
    return tuple(continuation_items)


def _to_plain_data(value: object) -> object:
    """Recursively remove SDK objects while preserving JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        plain_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMError("Provider data contained a non-string object key.")
            plain_mapping[key] = _to_plain_data(item)
        return plain_mapping
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_data(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain_data(model_dump(mode="json"))
        except Exception:
            raise LLMError("Could not serialize a provider output item.") from None
    raise LLMError("Could not serialize a provider output item.")


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
