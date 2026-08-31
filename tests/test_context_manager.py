"""Tests for protocol-safe, deterministic model-input context management."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import context_manager as context_module
from context_manager import ContextManager


STATIC_HISTORY = [
    {"role": "system", "content": "global system prompt"},
    {"role": "system", "content": "Project instructions: 保留 Unicode"},
    {"role": "user", "content": "original task"},
]


def function_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
    }


def function_output(
    call_id: str,
    *,
    success: bool = True,
    output: str = "result",
    error: str | None = None,
) -> dict[str, object]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(
            {"success": success, "output": output, "error": error},
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def state(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "original_goal": "original task",
        "workspace_changed": False,
        "changes_since_verification": False,
        "last_verification_command": None,
        "verification_evidence": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def add_step(
    manager: ContextManager,
    history: list[dict[str, object]],
    items: list[dict[str, object]],
) -> None:
    history.extend(items)
    manager.record_completed_step(len(history))


def result_for(context: list[dict[str, object]], call_id: str) -> dict[str, object]:
    item = next(
        candidate
        for candidate in context
        if candidate.get("type") == "function_call_output"
        and candidate.get("call_id") == call_id
    )
    output = item["output"]
    assert isinstance(output, str)
    parsed = json.loads(output)
    assert isinstance(parsed, dict)
    return parsed


def controller_message(
    context: list[dict[str, object]], heading: str
) -> str | None:
    for item in context:
        content = item.get("content")
        if item.get("role") == "system" and isinstance(content, str):
            if content.startswith(heading):
                return content
    return None


def test_short_history_is_copied_without_semantic_change() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=len(history))

    context = manager.build_context(history)

    assert context == history
    assert context is not history


def test_static_system_project_and_original_task_survive_compaction() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    add_step(
        manager,
        history,
        [
            function_call("old", "read_file", {"path": "大文件.py"}),
            function_output("old", output="内容" * 2_000),
        ],
    )
    add_step(manager, history, [{"role": "assistant", "content": "recent"}])

    context = manager.build_context(history)

    assert context[:3] == STATIC_HISTORY


def test_old_output_compacts_but_recent_steps_remain_fully_raw() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=2)
    raw_outputs: dict[str, str] = {}
    for index in range(5):
        call_id = f"call_{index}"
        raw_outputs[call_id] = f"完整输出 {index} " + "x" * 2_000
        add_step(
            manager,
            history,
            [
                function_call(call_id, "read_file", {"path": f"文件{index}.py"}),
                function_output(call_id, output=raw_outputs[call_id]),
            ],
        )

    context = manager.build_context(history)

    assert "old tool output compacted" in result_for(context, "call_0")["output"]
    assert result_for(context, "call_3")["output"] == raw_outputs["call_3"]
    assert result_for(context, "call_4")["output"] == raw_outputs["call_4"]


def test_long_history_model_view_is_meaningfully_smaller_and_deterministic() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    for index in range(8):
        add_step(
            manager,
            history,
            [
                function_call(f"call_{index}", "read_file", {"path": f"f{index}.py"}),
                function_output(f"call_{index}", output="数据" * 4_000),
            ],
        )

    first = manager.build_context(history)
    second = manager.build_context(history)

    assert first == second
    assert len(json.dumps(first, ensure_ascii=False)) < len(
        json.dumps(history, ensure_ascii=False)
    ) / 2
    assert "数据" in json.dumps(first, ensure_ascii=False)


def test_non_plain_sdk_like_objects_are_rejected_instead_of_leaking() -> None:
    class FakeSDKObject:
        pass

    manager = ContextManager(static_item_count=1)
    history = [{"role": "system", "content": FakeSDKObject()}]

    with pytest.raises(ValueError, match="plain JSON"):
        manager.build_context(history)


def test_call_output_linkage_and_multi_tool_mapping_remain_exact() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    add_step(
        manager,
        history,
        [
            function_call("a", "read_file", {"path": "a.py"}),
            function_call("b", "read_file", {"path": "b.py"}),
            function_output("a", output="A" * 1_000),
            function_output("b", output="B" * 1_000),
        ],
    )
    add_step(manager, history, [{"role": "assistant", "content": "later"}])

    context = manager.build_context(history)
    calls = {
        item["call_id"]: item
        for item in context
        if item.get("type") == "function_call"
    }
    outputs = {
        item["call_id"]: item
        for item in context
        if item.get("type") == "function_call_output"
    }

    assert set(calls) == {"a", "b"}
    assert set(outputs) == set(calls)
    assert json.loads(calls["a"]["arguments"])["path"] == "a.py"
    assert json.loads(calls["b"]["arguments"])["path"] == "b.py"
    assert result_for(context, "a")["output"].startswith("A")
    assert result_for(context, "b")["output"].startswith("B")


def test_unpaired_or_malformed_outputs_are_never_compacted_or_reassigned() -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    orphan = function_output("orphan", output="must remain")
    malformed = {
        "type": "function_call_output",
        "call_id": "bad",
        "output": "not-json",
    }
    add_step(manager, history, [orphan, malformed])
    add_step(manager, history, [{"role": "assistant", "content": "later"}])

    context = manager.build_context(history)

    assert orphan in context
    assert malformed in context


def test_reasoning_and_assistant_output_items_are_preserved_unmodified() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "reasoning_1",
        "summary": [{"type": "summary_text", "text": "synthetic structural marker"}],
    }
    assistant_output = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "visible text"}],
    }
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    add_step(
        manager,
        history,
        [
            reasoning,
            assistant_output,
            function_call("old", "read_file", {"path": "a.py"}),
            function_output("old", output="x" * 2_000),
        ],
    )
    add_step(manager, history, [{"role": "assistant", "content": "later"}])

    context = manager.build_context(history)

    assert reasoning in context
    assert assistant_output in context


@pytest.mark.parametrize(
    ("name", "arguments", "success", "expected"),
    [
        ("read_file", {"path": "app/服务.py"}, True, "Read file: app/服务.py"),
        (
            "search_text",
            {"query": "normalize_name", "path": "."},
            True,
            'Searched for "normalize_name" under .',
        ),
        ("write_file", {"path": "new.py", "content": "x"}, True, "Wrote file: new.py"),
        (
            "edit_file",
            {"path": "app.py", "old_text": "a", "new_text": "b"},
            True,
            "Modified file: app.py",
        ),
        ("run_command", {"command": "project-check"}, False, "Command failed: project-check"),
        ("run_command", {"command": "project-check"}, True, "Command succeeded: project-check"),
    ],
)
def test_summary_contains_only_observable_tool_facts(
    name: str,
    arguments: dict[str, object],
    success: bool,
    expected: str,
) -> None:
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    add_step(
        manager,
        history,
        [
            function_call("old", name, arguments),
            function_output("old", success=success, error="failed" if not success else None),
        ],
    )
    add_step(manager, history, [{"role": "assistant", "content": "later"}])

    context = manager.build_context(history)
    summary = controller_message(context, "Observable facts")

    assert summary is not None
    assert expected in summary
    assert "intended" not in summary.casefold()
    assert "reasoning" not in summary.casefold()


def test_context_summary_is_bounded_and_contains_no_reasoning_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_module, "MAX_CONTEXT_SUMMARY_CHARS", 180)
    history = deepcopy(STATIC_HISTORY)
    manager = ContextManager(static_item_count=3, recent_raw_steps=1)
    for index in range(20):
        add_step(
            manager,
            history,
            [
                function_call(
                    f"call_{index}", "read_file", {"path": f"long/path/{index}.py"}
                ),
                function_output(f"call_{index}", output="visible result"),
            ],
        )
    add_step(
        manager,
        history,
        [{"type": "reasoning", "id": "r", "summary": "SYNTHETIC-HIDDEN-TEXT"}],
    )

    context = manager.build_context(history)
    summary = controller_message(context, "Observable facts")

    assert summary is not None
    assert len(summary) <= len(context_module._SUMMARY_HEADING) + 180
    assert "SYNTHETIC-HIDDEN-TEXT" not in summary


def test_task_state_view_shows_stale_and_fresh_verification_without_mutation() -> None:
    manager = ContextManager(static_item_count=3)
    stale_state = state(
        workspace_changed=True,
        changes_since_verification=True,
        last_verification_command="cargo test",
        verification_evidence="previous pass",
    )
    snapshot = deepcopy(vars(stale_state))

    stale_context = manager.build_context(STATIC_HISTORY, stale_state)
    stale_message = controller_message(stale_context, "Controller task state")

    assert stale_message is not None
    assert "Unverified changes: yes" in stale_message
    assert "Last successful verification command: cargo test" in stale_message
    assert "Latest verification is fresh: no" in stale_message
    assert vars(stale_state) == snapshot

    fresh_state = state(
        workspace_changed=True,
        changes_since_verification=False,
        last_verification_command="cargo test",
        verification_evidence="pass",
    )
    fresh_context = manager.build_context(STATIC_HISTORY, fresh_state)
    fresh_message = controller_message(fresh_context, "Controller task state")
    assert fresh_message is not None
    assert "Unverified changes: no" in fresh_message
    assert "Latest verification is fresh: yes" in fresh_message
