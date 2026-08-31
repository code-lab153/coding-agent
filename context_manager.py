"""Build a bounded model-input view without mutating canonical Agent history."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Protocol

from llm import InputItem


# Preserve five complete model/tool iterations; compact only older completed work.
RECENT_RAW_STEPS = 5
# This is a soft provider-independent target. Protocol items win over forced truncation.
MAX_CONTEXT_CHARS = 60_000
MAX_CONTEXT_SUMMARY_CHARS = 4_000
_MAX_COMPACT_RESULT_CHARS = 240
_MIN_RESULT_CHARS_TO_COMPACT = 600
_MAX_STATE_TEXT_CHARS = 1_200
_SUMMARY_HEADING = "Observable facts from older completed tool calls:\n"
_STATE_HEADING = "Controller task state:\n"


class TaskStateView(Protocol):
    """The objective TaskState fields needed for a controller context view."""

    original_goal: str
    workspace_changed: bool
    changes_since_verification: bool
    last_verification_command: str | None
    verification_evidence: str | None


class ContextManager:
    """Create safe model input while retaining a complete canonical history."""

    def __init__(
        self,
        static_item_count: int,
        recent_raw_steps: int = RECENT_RAW_STEPS,
    ) -> None:
        if static_item_count < 0:
            raise ValueError("static_item_count cannot be negative")
        if recent_raw_steps < 1:
            raise ValueError("recent_raw_steps must be positive")
        self._static_item_count = static_item_count
        self._recent_raw_steps = recent_raw_steps
        self._completed_step_ends: list[int] = []

    def record_completed_step(self, canonical_history_length: int) -> None:
        """Record an iteration boundary after its complete outputs are appended."""
        if canonical_history_length < self._static_item_count:
            raise ValueError("completed step ends before static context")
        if (
            self._completed_step_ends
            and canonical_history_length < self._completed_step_ends[-1]
        ):
            raise ValueError("completed step boundaries must be monotonic")
        self._completed_step_ends.append(canonical_history_length)

    def build_context(
        self,
        canonical_history: Sequence[InputItem],
        task_state: TaskStateView | None = None,
    ) -> list[InputItem]:
        """Return a plain copied view with only old paired tool results compacted."""
        history = _plain_history_copy(canonical_history)
        if len(history) < self._static_item_count:
            raise ValueError("canonical history is shorter than its static context")

        compact_before = self._compaction_boundary()
        facts: list[str] = []
        if compact_before is not None:
            facts = _compact_old_tool_outputs(history, compact_before)

        controller_items = _controller_context_items(facts, task_state)
        if controller_items:
            history[self._static_item_count : self._static_item_count] = controller_items
            history = _trim_summary_to_soft_budget(history)
        return history

    def _compaction_boundary(self) -> int | None:
        """Return the exclusive history index older than the raw-step window."""
        if len(self._completed_step_ends) <= self._recent_raw_steps:
            return None
        return self._completed_step_ends[-self._recent_raw_steps - 1]


def _plain_history_copy(history: Sequence[InputItem]) -> list[InputItem]:
    """Copy already serialized input items and reject non-plain SDK objects."""
    copied = deepcopy(list(history))
    try:
        serialized = json.dumps(copied, ensure_ascii=False)
        plain = json.loads(serialized)
    except (TypeError, ValueError):
        raise ValueError("canonical history must contain only plain JSON data") from None
    if not isinstance(plain, list) or not all(isinstance(item, dict) for item in plain):
        raise ValueError("canonical history must contain input-item dictionaries")
    return plain


def _compact_old_tool_outputs(history: list[InputItem], boundary: int) -> list[str]:
    """Compact only old outputs with an earlier matching function call."""
    calls: dict[str, InputItem] = {}
    facts: list[str] = []
    for index, item in enumerate(history):
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type == "function_call" and isinstance(call_id, str):
            calls[call_id] = item
            continue
        if (
            item_type != "function_call_output"
            or index >= boundary
            or not isinstance(call_id, str)
        ):
            continue

        call = calls.get(call_id)
        parsed_result = _parse_tool_result(item.get("output"))
        if call is None or parsed_result is None:
            continue
        fact = _observable_fact(call, parsed_result)
        if fact is not None:
            facts.append(fact)
        raw_output = item.get("output")
        if isinstance(raw_output, str) and len(raw_output) > _MIN_RESULT_CHARS_TO_COMPACT:
            item["output"] = _compact_result_json(parsed_result)
    return facts


def _parse_tool_result(value: object) -> dict[str, object] | None:
    """Recognize the application-generated ToolResult JSON shape."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("success"), bool):
        return None
    return parsed


def _compact_result_json(result: Mapping[str, object]) -> str:
    """Preserve result status and concise evidence in deterministic JSON."""
    success = bool(result["success"])
    output = result.get("output")
    error = result.get("error")
    concise_output = _bounded_text(output, _MAX_COMPACT_RESULT_CHARS)
    concise_error = _bounded_text(error, _MAX_COMPACT_RESULT_CHARS)
    if concise_output:
        concise_output += "\n... [old tool output compacted]"
    else:
        concise_output = "[old tool output compacted]"
    return json.dumps(
        {"success": success, "output": concise_output, "error": concise_error or None},
        ensure_ascii=False,
        sort_keys=True,
    )


def _bounded_text(value: object, limit: int) -> str:
    """Return a deterministic bounded string without coercing arbitrary objects."""
    if not isinstance(value, str):
        return ""
    if len(value) <= limit:
        return value
    return value[:limit]


def _observable_fact(
    call: Mapping[str, object], result: Mapping[str, object]
) -> str | None:
    """Describe only tool name, arguments, and observed success/failure."""
    name = call.get("name")
    arguments_text = call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments_text, str):
        return None
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    success = result.get("success") is True

    if name == "read_file":
        return _path_fact("Read file", arguments)
    if name == "search_text":
        query = arguments.get("query")
        path = arguments.get("path", ".")
        if isinstance(query, str) and isinstance(path, str):
            return f"Searched for {json.dumps(query, ensure_ascii=False)} under {path}"
    if name == "write_file" and success:
        return _path_fact("Wrote file", arguments)
    if name == "edit_file" and success:
        return _path_fact("Modified file", arguments)
    if name == "run_command":
        command = arguments.get("command")
        if isinstance(command, str):
            status = "succeeded" if success else "failed"
            return f"Command {status}: {command}"
    if name == "list_files":
        path = arguments.get("path", ".")
        if isinstance(path, str):
            return f"Listed files under: {path}"
    return None


def _path_fact(prefix: str, arguments: Mapping[str, object]) -> str | None:
    path = arguments.get("path")
    return f"{prefix}: {path}" if isinstance(path, str) else None


def _controller_context_items(
    facts: Sequence[str], task_state: TaskStateView | None
) -> list[InputItem]:
    """Build distinct system observations without impersonating the user."""
    items: list[InputItem] = []
    if facts:
        summary = _bounded_summary(facts, MAX_CONTEXT_SUMMARY_CHARS)
        items.append({"role": "system", "content": _SUMMARY_HEADING + summary})
    if task_state is not None and (
        task_state.workspace_changed or task_state.last_verification_command is not None
    ):
        fresh = (
            task_state.last_verification_command is not None
            and not task_state.changes_since_verification
        )
        command = task_state.last_verification_command or "none"
        state_text = (
            f"Workspace changed: {'yes' if task_state.workspace_changed else 'no'}\n"
            f"Unverified changes: {'yes' if task_state.changes_since_verification else 'no'}\n"
            f"Last successful verification command: {command}\n"
            f"Latest verification is fresh: {'yes' if fresh else 'no'}"
        )
        items.append(
            {
                "role": "system",
                "content": _STATE_HEADING
                + _bounded_text(state_text, _MAX_STATE_TEXT_CHARS),
            }
        )
    return items


def _bounded_summary(facts: Sequence[str], limit: int) -> str:
    """Keep the newest complete observable facts within a fixed character bound."""
    selected: list[str] = []
    used = 0
    for fact in reversed(facts):
        line = f"- {fact}"
        added = len(line) + (1 if selected else 0)
        if used + added > limit:
            break
        selected.append(line)
        used += added
    selected.reverse()
    summary = "\n".join(selected)
    if len(selected) < len(facts):
        marker = "... [older observable facts omitted]\n"
        available = max(0, limit - len(marker))
        tail = summary[-available:] if available else ""
        summary = marker + tail
    return summary[:limit]


def _trim_summary_to_soft_budget(history: list[InputItem]) -> list[InputItem]:
    """Drop only the derived summary if it alone would exceed the soft target."""
    size = len(json.dumps(history, ensure_ascii=False))
    if size <= MAX_CONTEXT_CHARS:
        return history
    return [
        item
        for item in history
        if not (
            item.get("role") == "system"
            and isinstance(item.get("content"), str)
            and item["content"].startswith(_SUMMARY_HEADING)
        )
    ]
