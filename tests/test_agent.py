"""Network-free tests for the synchronous Agent controller."""

from copy import deepcopy
import json

import pytest

from agent import Agent, AgentError
from llm import LLMError, LLMResponse, ToolCall
from prompts import SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, ToolResult


class FakeLLMClient:
    """Return a predetermined response sequence and record complete inputs."""

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def send(
        self,
        messages: list[dict[str, object]],
        tools: tuple[dict[str, object], ...] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": deepcopy(messages), "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(
    call_id: str,
    name: str,
    arguments: dict[str, object],
    *,
    text: str = "",
    extra_items: tuple[dict[str, object], ...] = (),
) -> LLMResponse:
    """Create one self-contained fake model tool-call response."""
    items = list(extra_items)
    if text:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    items.append(
        {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
        }
    )
    return LLMResponse(
        text=text,
        tool_calls=(ToolCall(call_id, name, arguments),),
        continuation_items=tuple(items),
    )


def output_items(history: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return only function-call outputs from a recorded history."""
    return [item for item in history if item.get("type") == "function_call_output"]


def decoded_result(item: dict[str, object]) -> dict[str, object]:
    """Decode one deterministic ToolResult payload."""
    output = item["output"]
    assert isinstance(output, str)
    decoded = json.loads(output)
    assert isinstance(decoded, dict)
    return decoded


def test_direct_final_response_stops_immediately() -> None:
    fake_llm = FakeLLMClient([LLMResponse("Finished.")])

    result = Agent(fake_llm).run("Explain the project")

    assert result == "Finished."
    assert len(fake_llm.calls) == 1


def test_one_tool_call_then_final_response() -> None:
    executed: list[str] = []

    def read_file(path: str) -> ToolResult:
        executed.append(path)
        return ToolResult(True, "file contents")

    fake_llm = FakeLLMClient(
        [
            tool_response("call_read", "read_file", {"path": "main.py"}),
            LLMResponse("The file was read."),
        ]
    )

    result = Agent(fake_llm, tool_handlers={"read_file": read_file}).run(
        "Read main.py"
    )

    assert result == "The file was read."
    assert executed == ["main.py"]
    assert len(fake_llm.calls) == 2


def test_call_id_and_tool_output_are_preserved_in_next_history() -> None:
    def read_file(path: str) -> ToolResult:
        return ToolResult(True, f"contents of {path}")

    fake_llm = FakeLLMClient(
        [
            tool_response("call_exact_123", "read_file", {"path": "main.py"}),
            LLMResponse("Done."),
        ]
    )

    Agent(fake_llm, tool_handlers={"read_file": read_file}).run("Read main.py")

    items = output_items(fake_llm.calls[1]["messages"])
    assert items[0]["call_id"] == "call_exact_123"
    assert decoded_result(items[0]) == {
        "success": True,
        "output": "contents of main.py",
        "error": None,
    }


def test_system_prompt_and_original_task_remain_in_history() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "list_files", {}),
            LLMResponse("Done."),
        ]
    )

    Agent(fake_llm, tool_handlers={"list_files": lambda: ToolResult(True, "a.py")}).run(
        "Inspect the project"
    )

    second_history = fake_llm.calls[1]["messages"]
    assert second_history[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert second_history[1] == {"role": "user", "content": "Inspect the project"}


def test_failed_tool_result_is_feedback_and_agent_continues() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("call_missing", "read_file", {"path": "missing.py"}),
            LLMResponse("The file does not exist."),
        ]
    )
    handler = lambda path: ToolResult(False, error=f"File does not exist: {path}")

    result = Agent(fake_llm, tool_handlers={"read_file": handler}).run("Read it")

    assert result == "The file does not exist."
    payload = decoded_result(output_items(fake_llm.calls[1]["messages"])[0])
    assert payload["success"] is False
    assert "does not exist" in payload["error"]


def test_unknown_tool_is_returned_as_failure() -> None:
    fake_llm = FakeLLMClient(
        [tool_response("call_unknown", "delete_everything", {}), LLMResponse("Stopped.")]
    )

    result = Agent(fake_llm, tool_handlers={}).run("Try an unknown tool")

    assert result == "Stopped."
    payload = decoded_result(output_items(fake_llm.calls[1]["messages"])[0])
    assert payload["success"] is False
    assert payload["error"] == "Unknown tool: delete_everything"


@pytest.mark.parametrize(
    ("arguments", "error_text"),
    [
        ({}, "Missing required argument"),
        ({"path": 7}, "must be a string"),
        ({"path": "main.py", "extra": "no"}, "Unsupported argument"),
    ],
)
def test_invalid_arguments_do_not_call_handler(
    arguments: dict[str, object], error_text: str
) -> None:
    executed = False

    def forbidden_handler(**_arguments: object) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(True)

    fake_llm = FakeLLMClient(
        [tool_response("call_bad", "read_file", arguments), LLMResponse("Recovered.")]
    )

    Agent(fake_llm, tool_handlers={"read_file": forbidden_handler}).run("Read a file")

    payload = decoded_result(output_items(fake_llm.calls[1]["messages"])[0])
    assert not executed
    assert payload["success"] is False
    assert error_text in payload["error"]


def test_unexpected_handler_exception_becomes_failed_tool_result() -> None:
    def broken_handler(path: str) -> ToolResult:
        raise RuntimeError(f"sensitive details for {path}")

    fake_llm = FakeLLMClient(
        [
            tool_response("call_broken", "read_file", {"path": "main.py"}),
            LLMResponse("Recovered."),
        ]
    )

    result = Agent(fake_llm, tool_handlers={"read_file": broken_handler}).run("Read it")

    assert result == "Recovered."
    payload = decoded_result(output_items(fake_llm.calls[1]["messages"])[0])
    assert payload["success"] is False
    assert payload["error"] == "Tool read_file failed unexpectedly."
    assert "sensitive" not in payload["error"]


def test_multiple_tool_calls_execute_sequentially_with_matching_ids() -> None:
    execution_order: list[str] = []
    calls = (
        ToolCall("call_1", "read_file", {"path": "a.py"}),
        ToolCall("call_2", "read_file", {"path": "b.py"}),
    )
    continuation = tuple(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        }
        for call in calls
    )
    fake_llm = FakeLLMClient(
        [LLMResponse("", calls, continuation), LLMResponse("Both read.")]
    )

    def read_file(path: str) -> ToolResult:
        execution_order.append(path)
        return ToolResult(True, path)

    Agent(fake_llm, tool_handlers={"read_file": read_file}).run("Read both")

    assert execution_order == ["a.py", "b.py"]
    outputs = output_items(fake_llm.calls[1]["messages"])
    assert [item["call_id"] for item in outputs] == ["call_1", "call_2"]


def test_text_plus_tool_call_does_not_terminate_early() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "list_files", {}, text="I will inspect first."),
            LLMResponse("Inspection complete."),
        ]
    )

    result = Agent(
        fake_llm,
        tool_handlers={"list_files": lambda: ToolResult(True, "main.py")},
    ).run("Inspect")

    assert result == "Inspection complete."
    assert len(fake_llm.calls) == 2


def test_repeated_tool_calls_hit_max_steps() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "list_files", {}),
            tool_response("call_2", "list_files", {}),
        ]
    )

    with pytest.raises(AgentError, match="maximum of 2 steps"):
        Agent(
            fake_llm,
            tool_handlers={"list_files": lambda: ToolResult(True, "main.py")},
            max_steps=2,
        ).run("Never finish")

    assert len(fake_llm.calls) == 2


def test_fatal_llm_error_becomes_agent_error() -> None:
    fake_llm = FakeLLMClient([LLMError("provider failure")])

    with pytest.raises(AgentError, match="model communication failed"):
        Agent(fake_llm).run("Do work")


@pytest.mark.parametrize("task", ["", "   ", None])
def test_empty_or_invalid_task_is_rejected(task: object) -> None:
    fake_llm = FakeLLMClient([LLMResponse("unused")])

    with pytest.raises(AgentError, match="non-empty string"):
        Agent(fake_llm).run(task)

    assert fake_llm.calls == []


def test_reasoning_continuation_items_are_preserved_in_history() -> None:
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "content": []}
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_1", "list_files", {}, extra_items=(reasoning,)
            ),
            LLMResponse("Done."),
        ]
    )

    Agent(
        fake_llm,
        tool_handlers={"list_files": lambda: ToolResult(True, "main.py")},
    ).run("Inspect")

    assert reasoning in fake_llm.calls[1]["messages"]


def test_system_prompt_sets_role_evidence_and_verification_expectations() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "coding agent" in prompt
    assert "evidence" in prompt
    assert "verify" in prompt
