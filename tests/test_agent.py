"""Network-free tests for the synchronous Agent controller."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import agent as agent_module
import config
from agent import Agent, AgentError, TaskState
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


def multiple_tool_response(calls: tuple[ToolCall, ...]) -> LLMResponse:
    """Create one fake model response containing several ordered tool calls."""
    continuation = tuple(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        }
        for call in calls
    )
    return LLMResponse("", calls, continuation)


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


def test_trace_shows_step_tool_arguments_and_successful_result() -> None:
    trace: list[str] = []
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "read_file", {"path": "main.py"}),
            LLMResponse("Done."),
        ]
    )

    Agent(
        fake_llm,
        tool_handlers={"read_file": lambda path: ToolResult(True, f"read {path}")},
        trace=trace.append,
    ).run("Read main.py")

    assert len(trace) == 1
    assert trace[0].splitlines()[0] == "[Step 1/20]"
    assert "Tool: read_file" in trace[0]
    assert '"path": "main.py"' in trace[0]
    assert "success: true" in trace[0]
    assert "read main.py" in trace[0]


def test_trace_shows_failure_but_tool_failure_does_not_stop_agent() -> None:
    trace: list[str] = []
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "run_command", {"command": "pytest"}),
            LLMResponse("I recovered from the failure."),
        ]
    )

    result = Agent(
        fake_llm,
        tool_handlers={
            "run_command": lambda command: ToolResult(
                False,
                output=f"{command}: one test failed",
                error="Command exited with return code 1.",
            )
        },
        trace=trace.append,
    ).run("Fix tests")

    assert result == "I recovered from the failure."
    assert "success: false" in trace[0]
    assert "one test failed" in trace[0]
    assert "return code 1" in trace[0]
    assert len(fake_llm.calls) == 2


def test_multi_tool_trace_indexes_calls_without_consuming_extra_steps() -> None:
    trace: list[str] = []
    execution_order: list[str] = []
    calls = (
        ToolCall("call_1", "read_file", {"path": "a.py"}),
        ToolCall("call_2", "read_file", {"path": "b.py"}),
        ToolCall("call_3", "read_file", {"path": "c.py"}),
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
        [LLMResponse("", calls, continuation), LLMResponse("Done.")]
    )

    def read_file(path: str) -> ToolResult:
        execution_order.append(path)
        return ToolResult(True, path)

    result = Agent(
        fake_llm,
        tool_handlers={"read_file": read_file},
        max_steps=2,
        trace=trace.append,
    ).run("Read three files")

    assert result == "Done."
    assert execution_order == ["a.py", "b.py", "c.py"]
    assert len(fake_llm.calls) == 2
    assert len(trace) == 3
    assert "[Step 1/2 - Tool 1/3]" in trace[0]
    assert "[Step 1/2 - Tool 2/3]" in trace[1]
    assert "[Step 1/2 - Tool 3/3]" in trace[2]


def test_trace_redacts_secrets_and_never_prints_reasoning_items() -> None:
    trace: list[str] = []
    private_reasoning = "hidden private reasoning must not appear"
    command = "echo MODEL_API_KEY=super-secret-value"
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_1",
                "run_command",
                {"command": command},
                extra_items=(
                    {
                        "type": "reasoning",
                        "summary": [private_reasoning],
                    },
                ),
            ),
            LLMResponse("Done."),
        ]
    )

    Agent(
        fake_llm,
        tool_handlers={
            "run_command": lambda command: ToolResult(True, output=command)
        },
        trace=trace.append,
    ).run("Run a command")

    rendered = "\n".join(trace)
    assert "super-secret-value" not in rendered
    assert private_reasoning not in rendered
    assert "[redacted]" in rendered


def test_initial_task_state_contains_original_goal() -> None:
    agent = Agent(FakeLLMClient([LLMResponse("Done.")]))

    result = agent.run("Inspect the project")

    assert result == "Done."
    assert agent.task_state == TaskState(original_goal="Inspect the project")


@pytest.mark.parametrize(
    ("name", "arguments", "handler"),
    [
        (
            "write_file",
            {"path": "item.py", "content": "new"},
            lambda path, content: ToolResult(True, f"wrote {path}: {content}"),
        ),
        (
            "edit_file",
            {"path": "item.py", "old_text": "old", "new_text": "new"},
            lambda path, old_text, new_text: ToolResult(
                True, f"edited {path}: {old_text} -> {new_text}"
            ),
        ),
    ],
)
def test_successful_file_change_marks_workspace_changed_and_verification_stale(
    name: str,
    arguments: dict[str, object],
    handler: object,
) -> None:
    fake_llm = FakeLLMClient([tool_response("call_change", name, arguments)])
    agent = Agent(fake_llm, tool_handlers={name: handler}, max_steps=1)

    with pytest.raises(AgentError, match="maximum of 1 step"):
        agent.run("Change a file")

    assert agent.task_state is not None
    assert agent.task_state.workspace_changed
    assert agent.task_state.changes_since_verification


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("write_file", {"path": "item.py", "content": "new"}),
        (
            "edit_file",
            {"path": "item.py", "old_text": "old", "new_text": "new"},
        ),
    ],
)
def test_failed_file_change_does_not_mark_workspace_changed(
    name: str, arguments: dict[str, object]
) -> None:
    fake_llm = FakeLLMClient([tool_response("call_change", name, arguments)])
    agent = Agent(
        fake_llm,
        tool_handlers={name: lambda **_arguments: ToolResult(False, error="failed")},
        max_steps=1,
    )

    with pytest.raises(AgentError, match="maximum of 1 step"):
        agent.run("Try a change")

    assert agent.task_state is not None
    assert not agent.task_state.workspace_changed
    assert not agent.task_state.changes_since_verification


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("list_files", {}),
        ("read_file", {"path": "item.py"}),
        ("search_text", {"query": "target"}),
    ],
)
def test_read_only_tools_do_not_mark_workspace_changed(
    name: str, arguments: dict[str, object]
) -> None:
    fake_llm = FakeLLMClient([tool_response("call_read", name, arguments)])
    agent = Agent(
        fake_llm,
        tool_handlers={name: lambda **_arguments: ToolResult(True, "observed")},
        max_steps=1,
    )

    with pytest.raises(AgentError, match="maximum of 1 step"):
        agent.run("Inspect")

    assert agent.task_state is not None
    assert not agent.task_state.workspace_changed
    assert not agent.task_state.changes_since_verification


def test_later_modification_makes_successful_verification_stale_again() -> None:
    calls = (
        ToolCall("call_write", "write_file", {"path": "item.py", "content": "one"}),
        ToolCall("call_check", "run_command", {"command": "custom-check"}),
        ToolCall(
            "call_edit",
            "edit_file",
            {"path": "item.py", "old_text": "one", "new_text": "two"},
        ),
    )
    fake_llm = FakeLLMClient([multiple_tool_response(calls)])
    agent = Agent(
        fake_llm,
        tool_handlers={
            "write_file": lambda **_arguments: ToolResult(True, "written"),
            "run_command": lambda command: ToolResult(True, f"{command} passed"),
            "edit_file": lambda **_arguments: ToolResult(True, "edited"),
        },
        max_steps=1,
    )

    with pytest.raises(AgentError, match="maximum of 1 step"):
        agent.run("Change and verify")

    assert agent.task_state is not None
    assert agent.task_state.workspace_changed
    assert agent.task_state.changes_since_verification
    assert agent.task_state.last_verification_command == "custom-check"


def test_verification_gate_rejects_premature_final_and_adds_observation() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_write", "write_file", {"path": "item.py", "content": "new"}
            ),
            LLMResponse("Premature completion."),
            tool_response("call_check", "run_command", {"command": "project-check"}),
            LLMResponse("Verified completion."),
        ]
    )
    agent = Agent(
        fake_llm,
        tool_handlers={
            "write_file": lambda **_arguments: ToolResult(True, "written"),
            "run_command": lambda command: ToolResult(True, f"{command} passed"),
        },
    )

    result = agent.run("Change and verify")

    assert result == "Verified completion."
    assert len(fake_llm.calls) == 4
    controller_items = [
        item
        for item in fake_llm.calls[2]["messages"]
        if item.get("role") == "system"
        and "no successful verification" in str(item.get("content"))
    ]
    assert len(controller_items) == 1
    assert "workspace has changed" in controller_items[0]["content"]


def test_failed_run_command_does_not_satisfy_verification() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_write", "write_file", {"path": "item.py", "content": "new"}
            ),
            tool_response("call_check", "run_command", {"command": "project-check"}),
            LLMResponse("Done without passing checks."),
        ]
    )
    agent = Agent(
        fake_llm,
        tool_handlers={
            "write_file": lambda **_arguments: ToolResult(True, "written"),
            "run_command": lambda command: ToolResult(
                False, output=f"{command} failed", error="exit 1"
            ),
        },
        max_steps=3,
    )

    with pytest.raises(AgentError, match="maximum of 3 steps"):
        agent.run("Change and verify")

    assert agent.task_state is not None
    assert agent.task_state.changes_since_verification
    assert agent.task_state.last_verification_command is None


def test_successful_generic_command_after_change_allows_final_response() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_edit",
                "edit_file",
                {"path": "item.rs", "old_text": "old", "new_text": "new"},
            ),
            tool_response("call_check", "run_command", {"command": "cargo check"}),
            LLMResponse("Changed and checked."),
        ]
    )
    agent = Agent(
        fake_llm,
        tool_handlers={
            "edit_file": lambda **_arguments: ToolResult(True, "edited"),
            "run_command": lambda command: ToolResult(
                True, f"verification passed: {command}"
            ),
        },
    )

    result = agent.run("Update the project")

    assert result == "Changed and checked."
    assert agent.task_state is not None
    assert not agent.task_state.changes_since_verification
    assert agent.task_state.last_verification_command == "cargo check"
    assert "verification passed" in (agent.task_state.verification_evidence or "")


def test_successful_command_before_change_does_not_verify_later_modification() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("call_check", "run_command", {"command": "initial-check"}),
            tool_response(
                "call_write", "write_file", {"path": "item.py", "content": "new"}
            ),
            LLMResponse("Done."),
        ]
    )
    agent = Agent(
        fake_llm,
        tool_handlers={
            "run_command": lambda command: ToolResult(True, f"{command} passed"),
            "write_file": lambda **_arguments: ToolResult(True, "written"),
        },
        max_steps=3,
    )

    with pytest.raises(AgentError, match="maximum of 3 steps"):
        agent.run("Check then change")

    assert agent.task_state is not None
    assert agent.task_state.changes_since_verification
    assert agent.task_state.last_verification_command is None


def test_verification_gate_remains_bounded_by_max_steps() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response(
                "call_write", "write_file", {"path": "item.py", "content": "new"}
            ),
            LLMResponse("I will not verify."),
        ]
    )
    agent = Agent(
        fake_llm,
        tool_handlers={"write_file": lambda **_arguments: ToolResult(True, "written")},
        max_steps=2,
    )

    with pytest.raises(AgentError, match="maximum of 2 steps"):
        agent.run("Change without verification")

    assert len(fake_llm.calls) == 2


def test_third_consecutive_identical_call_is_blocked_with_original_call_id() -> None:
    executed: list[str] = []
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "read_file", {"path": "item.py"}),
            tool_response("call_2", "read_file", {"path": "item.py"}),
            tool_response("call_3", "read_file", {"path": "item.py"}),
            LLMResponse("Recovered."),
        ]
    )

    def read_file(path: str) -> ToolResult:
        executed.append(path)
        return ToolResult(True, "contents")

    result = Agent(fake_llm, tool_handlers={"read_file": read_file}).run("Read")

    assert result == "Recovered."
    assert executed == ["item.py", "item.py"]
    outputs = output_items(fake_llm.calls[3]["messages"])
    assert len(outputs) == 3
    assert outputs[2]["call_id"] == "call_3"
    repeated_result = decoded_result(outputs[2])
    assert repeated_result["success"] is False
    assert "Repeated identical tool call" in repeated_result["error"]


def test_agent_can_recover_with_different_action_after_repeat_feedback() -> None:
    read_count = 0
    list_count = 0
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "read_file", {"path": "item.py"}),
            tool_response("call_2", "read_file", {"path": "item.py"}),
            tool_response("call_3", "read_file", {"path": "item.py"}),
            tool_response("call_4", "list_files", {}),
            LLMResponse("Changed strategy."),
        ]
    )

    def read_file(path: str) -> ToolResult:
        nonlocal read_count
        read_count += 1
        return ToolResult(True, path)

    def list_files() -> ToolResult:
        nonlocal list_count
        list_count += 1
        return ToolResult(True, "item.py")

    result = Agent(
        fake_llm,
        tool_handlers={"read_file": read_file, "list_files": list_files},
    ).run("Recover")

    assert result == "Changed strategy."
    assert read_count == 2
    assert list_count == 1


def test_different_action_resets_consecutive_repeat_count() -> None:
    read_count = 0
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "read_file", {"path": "item.py"}),
            tool_response("call_2", "read_file", {"path": "item.py"}),
            tool_response("call_3", "list_files", {}),
            tool_response("call_4", "read_file", {"path": "item.py"}),
            LLMResponse("Done."),
        ]
    )

    def read_file(path: str) -> ToolResult:
        nonlocal read_count
        read_count += 1
        return ToolResult(True, path)

    result = Agent(
        fake_llm,
        tool_handlers={
            "read_file": read_file,
            "list_files": lambda: ToolResult(True, "item.py"),
        },
    ).run("Inspect")

    assert result == "Done."
    assert read_count == 3


def test_same_read_before_and_after_edit_is_not_treated_as_repeat() -> None:
    read_count = 0
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "read_file", {"path": "item.py"}),
            tool_response(
                "call_2",
                "edit_file",
                {"path": "item.py", "old_text": "old", "new_text": "new"},
            ),
            tool_response("call_3", "read_file", {"path": "item.py"}),
            tool_response("call_4", "run_command", {"command": "project-check"}),
            LLMResponse("Done."),
        ]
    )

    def read_file(path: str) -> ToolResult:
        nonlocal read_count
        read_count += 1
        return ToolResult(True, path)

    result = Agent(
        fake_llm,
        tool_handlers={
            "read_file": read_file,
            "edit_file": lambda **_arguments: ToolResult(True, "edited"),
            "run_command": lambda command: ToolResult(True, f"{command} passed"),
        },
    ).run("Edit")

    assert result == "Done."
    assert read_count == 2


def test_repeated_command_separated_by_edit_is_not_treated_as_repeat() -> None:
    command_count = 0
    fake_llm = FakeLLMClient(
        [
            tool_response("call_1", "run_command", {"command": "python -m pytest"}),
            tool_response(
                "call_2",
                "edit_file",
                {"path": "item.py", "old_text": "old", "new_text": "new"},
            ),
            tool_response("call_3", "run_command", {"command": "python -m pytest"}),
            LLMResponse("Done."),
        ]
    )

    def run_command(command: str) -> ToolResult:
        nonlocal command_count
        command_count += 1
        return ToolResult(True, f"{command} passed")

    result = Agent(
        fake_llm,
        tool_handlers={
            "run_command": run_command,
            "edit_file": lambda **_arguments: ToolResult(True, "edited"),
        },
    ).run("Test, edit, test")

    assert result == "Done."
    assert command_count == 2


def test_repeat_detection_is_ordered_within_one_multi_tool_iteration() -> None:
    executed: list[str] = []
    calls = (
        ToolCall("call_1", "read_file", {"path": "item.py"}),
        ToolCall("call_2", "read_file", {"path": "item.py"}),
        ToolCall("call_3", "read_file", {"path": "item.py"}),
    )
    fake_llm = FakeLLMClient(
        [multiple_tool_response(calls), LLMResponse("Recovered.")]
    )

    def read_file(path: str) -> ToolResult:
        executed.append(path)
        return ToolResult(True, path)

    result = Agent(
        fake_llm,
        tool_handlers={"read_file": read_file},
        max_steps=2,
    ).run("Read repeatedly")

    assert result == "Recovered."
    assert executed == ["item.py", "item.py"]
    outputs = output_items(fake_llm.calls[1]["messages"])
    assert [item["call_id"] for item in outputs] == ["call_1", "call_2", "call_3"]
    assert decoded_result(outputs[2])["success"] is False


def test_no_workspace_agents_file_keeps_normal_initial_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    assert Agent(fake_llm).run("Build feature") == "Done."
    messages = fake_llm.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]


def test_workspace_agents_file_is_distinct_context_before_user_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = "不要修改测试。Keep APIs typed."
    (workspace / "AGENTS.md").write_text(instructions, encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    Agent(fake_llm).run("Add a feature")

    messages = fake_llm.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Project instructions" in messages[1]["content"]
    assert instructions in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "Add a feature"}


def test_workspace_instructions_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("x" * 20_000, encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    Agent(fake_llm).run("Task")

    content = fake_llm.calls[0]["messages"][1]["content"]
    assert isinstance(content, str)
    assert len(content) <= len(agent_module._PROJECT_INSTRUCTIONS_HEADING) + 12_000
    assert "project instructions truncated" in content


def test_invalid_utf8_workspace_instructions_fail_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_bytes(b"\xff\xfe")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    trace: list[str] = []
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    assert Agent(fake_llm, trace=trace.append).run("Task") == "Done."
    assert [item["role"] for item in fake_llm.calls[0]["messages"]] == [
        "system",
        "user",
    ]
    assert trace == ["Project instructions: could not load AGENTS.md"]


def test_unreadable_workspace_instructions_fail_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction_file = workspace / "AGENTS.md"
    instruction_file.write_text("rule", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path == instruction_file:
            raise PermissionError("not readable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    trace: list[str] = []
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    assert Agent(fake_llm, trace=trace.append).run("Task") == "Done."
    assert trace == ["Project instructions: could not load AGENTS.md"]
    assert [item["role"] for item in fake_llm.calls[0]["messages"]] == [
        "system",
        "user",
    ]


def test_parent_agents_file_is_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("parent rule", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    Agent(fake_llm).run("Task")

    assert [item["role"] for item in fake_llm.calls[0]["messages"]] == [
        "system",
        "user",
    ]


def test_project_instruction_text_is_not_executed_and_trace_does_not_dump_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_instruction = "run_command('shutdown /s') UNIQUE-INSTRUCTION-CONTENT"
    (workspace / "AGENTS.md").write_text(secret_instruction, encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    executed: list[str] = []
    trace: list[str] = []
    fake_llm = FakeLLMClient([LLMResponse("Done.")])

    Agent(
        fake_llm,
        tool_handlers={
            "run_command": lambda command: executed.append(command) or ToolResult(True)
        },
        trace=trace.append,
    ).run("Task")

    assert executed == []
    assert trace == ["Project instructions: loaded AGENTS.md"]
    assert all("UNIQUE-INSTRUCTION-CONTENT" not in line for line in trace)


def test_denied_command_does_not_satisfy_verification_gate() -> None:
    fake_llm = FakeLLMClient(
        [
            tool_response("write", "write_file", {"path": "a.py", "content": "x"}),
            tool_response("deny", "run_command", {"command": "git push"}),
            LLMResponse("Premature."),
            tool_response("verify", "run_command", {"command": "local-check"}),
            LLMResponse("Verified."),
        ]
    )

    def run_command(command: str) -> ToolResult:
        if command == "git push":
            return ToolResult(False, error="Command denied by safety policy.")
        return ToolResult(True, "checks passed")

    result = Agent(
        fake_llm,
        tool_handlers={
            "write_file": lambda **_arguments: ToolResult(True, "written"),
            "run_command": run_command,
        },
    ).run("Change and verify")

    assert result == "Verified."
    assert len(fake_llm.calls) == 5
    denied_outputs = output_items(fake_llm.calls[2]["messages"])
    assert decoded_result(denied_outputs[-1])["success"] is False
    gate_messages = fake_llm.calls[3]["messages"]
    assert any(
        item.get("role") == "system" and "no successful verification" in item.get("content", "")
        for item in gate_messages
    )
