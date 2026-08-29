"""Tests for the minimal, network-free LLM communication layer."""

from dataclasses import dataclass, field
from typing import Any

import pytest

import llm
from config import ConfigurationError
from llm import LLMClient, LLMError, LLMResponse, ToolCall
from tools import TOOL_HANDLERS, TOOL_SCHEMAS


@dataclass
class FakeProviderResponse:
    """Small stand-in for the provider response used by tests."""

    output_text: str
    output: list[object] = field(default_factory=list)


@dataclass
class FakeFunctionCall:
    """Stand in for one provider-native function_call output item."""

    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


class FakeResponsesAPI:
    """Record Responses API calls and return a configured result."""

    def __init__(self, response: object, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    """Expose the same responses attribute that LLMClient uses."""

    def __init__(self, responses: FakeResponsesAPI) -> None:
        self.responses = responses


def install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    response: object | None = None,
    error: Exception | None = None,
) -> tuple[dict[str, str], FakeResponsesAPI]:
    """Replace the SDK constructor without making a network request."""
    fake_responses = FakeResponsesAPI(
        FakeProviderResponse("hello") if response is None else response,
        error,
    )
    captured_options: dict[str, str] = {}

    def fake_openai(**kwargs: str) -> FakeOpenAIClient:
        captured_options.update(kwargs)
        return FakeOpenAIClient(fake_responses)

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    return captured_options, fake_responses


def set_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set obvious non-secret placeholders for configuration tests."""
    monkeypatch.setenv("MODEL_API_KEY", "test-placeholder")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)


def test_send_returns_normalized_text(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    _, fake_responses = install_fake_openai(
        monkeypatch, FakeProviderResponse("Hello from the model.")
    )
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
    ]

    result = LLMClient().send(messages)

    assert result == LLMResponse("Hello from the model.")
    assert fake_responses.calls == [{"model": "test-model", "input": messages}]


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_NAME", "test-model")

    with pytest.raises(ConfigurationError, match="MODEL_API_KEY is required"):
        LLMClient()


def test_missing_model_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "test-placeholder")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(ConfigurationError, match="MODEL_NAME is required"):
        LLMClient()


def test_optional_base_url_is_passed_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid/v1")
    captured_options, _ = install_fake_openai(monkeypatch)

    LLMClient()

    assert captured_options == {
        "api_key": "test-placeholder",
        "base_url": "https://example.invalid/v1",
    }


def test_absent_base_url_uses_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    captured_options, _ = install_fake_openai(monkeypatch)

    LLMClient()

    assert captured_options == {"api_key": "test-placeholder"}


def test_provider_exception_becomes_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    install_fake_openai(monkeypatch, error=RuntimeError("provider unavailable"))

    with pytest.raises(LLMError, match="Model request failed") as captured:
        LLMClient().send([{"role": "user", "content": "Hello."}])

    assert "test-placeholder" not in str(captured.value)


@pytest.mark.parametrize("response", [FakeProviderResponse(""), object()])
def test_empty_or_malformed_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> None:
    set_valid_environment(monkeypatch)
    install_fake_openai(monkeypatch, response=response)

    with pytest.raises(LLMError, match="did not contain text or tool calls"):
        LLMClient().send([{"role": "user", "content": "Hello."}])


def test_result_is_not_provider_object(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    provider_response = FakeProviderResponse("normalized")
    install_fake_openai(monkeypatch, response=provider_response)

    result = LLMClient().send([{"role": "user", "content": "Hello."}])

    assert isinstance(result, LLMResponse)
    assert result is not provider_response


def test_malformed_messages_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    install_fake_openai(monkeypatch)

    with pytest.raises(LLMError, match="unsupported role"):
        LLMClient().send([{"role": "tool", "content": "not supported yet"}])


def test_single_function_call_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_123", "read_file", '{"path":"main.py"}')
    install_fake_openai(monkeypatch, FakeProviderResponse("", [provider_call]))

    result = LLMClient().send([{"role": "user", "content": "Read main.py"}])

    assert result == LLMResponse(
        text="",
        tool_calls=(
            ToolCall("call_123", "read_file", {"path": "main.py"}),
        ),
    )


def test_multiple_function_calls_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    provider_calls = [
        FakeFunctionCall("call_1", "list_files", "{}"),
        FakeFunctionCall("call_2", "read_file", '{"path":"main.py"}'),
    ]
    install_fake_openai(monkeypatch, FakeProviderResponse("", provider_calls))

    result = LLMClient().send([{"role": "user", "content": "Inspect files"}])

    assert result.tool_calls == (
        ToolCall("call_1", "list_files", {}),
        ToolCall("call_2", "read_file", {"path": "main.py"}),
    )


def test_text_and_function_call_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_1", "list_files", "{}")
    install_fake_openai(
        monkeypatch, FakeProviderResponse("I will inspect the project.", [provider_call])
    )

    result = LLMClient().send([{"role": "user", "content": "Inspect files"}])

    assert result.text == "I will inspect the project."
    assert result.tool_calls == (ToolCall("call_1", "list_files", {}),)


def test_function_arguments_must_be_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_bad", "read_file", "{not-json}")
    install_fake_openai(monkeypatch, FakeProviderResponse("", [provider_call]))

    with pytest.raises(LLMError, match="invalid JSON arguments"):
        LLMClient().send([{"role": "user", "content": "Read a file"}])


def test_function_arguments_must_decode_to_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_bad", "read_file", '["main.py"]')
    install_fake_openai(monkeypatch, FakeProviderResponse("", [provider_call]))

    with pytest.raises(LLMError, match="must be a JSON object"):
        LLMClient().send([{"role": "user", "content": "Read a file"}])


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"type": "function_call", "name": "read_file", "arguments": "{}"}, "call_id"),
        (
            {
                "type": "function_call",
                "call_id": "   ",
                "name": "read_file",
                "arguments": "{}",
            },
            "call_id",
        ),
        ({"type": "function_call", "call_id": "call_1", "arguments": "{}"}, "name"),
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "   ",
                "arguments": "{}",
            },
            "name",
        ),
        ({"type": "function_call", "call_id": "call_1", "name": "read_file"}, "JSON text"),
    ],
)
def test_invalid_function_call_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch, item: dict[str, str], message: str
) -> None:
    set_valid_environment(monkeypatch)
    install_fake_openai(monkeypatch, FakeProviderResponse("", [item]))

    with pytest.raises(LLMError, match=message):
        LLMClient().send([{"role": "user", "content": "Read a file"}])


def test_tool_call_is_internal_type_not_provider_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_1", "read_file", '{"path":"main.py"}')
    install_fake_openai(monkeypatch, FakeProviderResponse("", [provider_call]))

    result = LLMClient().send([{"role": "user", "content": "Read main.py"}])

    assert isinstance(result.tool_calls[0], ToolCall)
    assert result.tool_calls[0] is not provider_call


def test_tools_are_passed_to_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    _, fake_responses = install_fake_openai(monkeypatch)
    messages = [{"role": "user", "content": "Inspect files"}]

    LLMClient().send(messages, tools=TOOL_SCHEMAS)

    assert fake_responses.calls == [
        {
            "model": "test-model",
            "input": messages,
            "tools": list(TOOL_SCHEMAS),
            "parallel_tool_calls": False,
        }
    ]


def test_empty_tools_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    _, fake_responses = install_fake_openai(monkeypatch)
    messages = [{"role": "user", "content": "Say hello"}]

    LLMClient().send(messages, tools=[])

    assert fake_responses.calls == [{"model": "test-model", "input": messages}]


def test_parsing_does_not_execute_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    set_valid_environment(monkeypatch)
    provider_call = FakeFunctionCall("call_1", "read_file", '{"path":"main.py"}')
    install_fake_openai(monkeypatch, FakeProviderResponse("", [provider_call]))
    executed = False

    def forbidden_handler(**_arguments: object) -> object:
        nonlocal executed
        executed = True
        raise AssertionError("Phase 4 must not execute tools")

    monkeypatch.setitem(TOOL_HANDLERS, "read_file", forbidden_handler)

    result = LLMClient().send([{"role": "user", "content": "Read main.py"}])

    assert result.tool_calls[0].name == "read_file"
    assert not executed
