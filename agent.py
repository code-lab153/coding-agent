"""Small synchronous Agent loop and local tool-dispatch boundary."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re

import config
from config import MAX_STEPS
from llm import InputItem, LLMClient, LLMError, ToolCall
from prompts import SYSTEM_PROMPT
from tools import TOOL_HANDLERS, TOOL_SCHEMAS, ToolResult


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:MODEL_API_KEY|API_KEY|TOKEN|PASSWORD|SECRET)\s*[:=]\s*)(\S+)"
)
_SECRET_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_REPEAT_THRESHOLD = 3
_VERIFICATION_EVIDENCE_LIMIT = 500
_MAX_PROJECT_INSTRUCTIONS = 12_000
_PROJECT_INSTRUCTIONS_HEADING = "Project instructions from workspace AGENTS.md:\n"
_PROJECT_INSTRUCTIONS_TRUNCATION = "\n... [project instructions truncated]"
_VERIFICATION_REQUIRED_OBSERVATION = (
    "Controller observation: the workspace has changed, but no successful "
    "verification command has run after the latest modification. Run an "
    "appropriate verification command before completing the task."
)
_REPEATED_ACTION_ERROR = (
    "Repeated identical tool call detected. Reconsider the current approach "
    "instead of repeating the same action."
)


class AgentError(RuntimeError):
    """Raised when the Agent controller cannot safely continue."""


@dataclass
class TaskState:
    """Objective execution facts for one Agent run, separate from conversation."""

    original_goal: str
    workspace_changed: bool = False
    changes_since_verification: bool = False
    last_verification_command: str | None = None
    verification_evidence: str | None = None


class Agent:
    """Coordinate model decisions, local tool actions, and conversation history."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_schemas: Sequence[dict[str, object]] = TOOL_SCHEMAS,
        tool_handlers: Mapping[str, Callable[..., ToolResult]] = TOOL_HANDLERS,
        max_steps: int = MAX_STEPS,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        """Store the explicit collaborators and finite loop budget."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise AgentError("max_steps must be a positive integer.")
        self._llm_client = llm_client
        self._tool_schemas = tuple(tool_schemas)
        self._tool_handlers = dict(tool_handlers)
        self._max_steps = max_steps
        self._trace = trace
        self.task_state: TaskState | None = None

    def run(self, task: str) -> str:
        """Run the decide-act-observe loop until text completion or a fatal error."""
        if not isinstance(task, str) or not task.strip():
            raise AgentError("Task must be a non-empty string.")

        state = TaskState(original_goal=task)
        self.task_state = state
        project_instructions, instruction_trace = _load_project_instructions()
        if self._trace is not None and instruction_trace is not None:
            self._trace(instruction_trace)

        history: list[InputItem] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if project_instructions is not None:
            history.append(
                {
                    "role": "system",
                    "content": _PROJECT_INSTRUCTIONS_HEADING + project_instructions,
                }
            )
        history.append({"role": "user", "content": task})
        previous_fingerprint: str | None = None
        consecutive_repeat_count = 0

        for step in range(1, self._max_steps + 1):
            try:
                response = self._llm_client.send(history, tools=self._tool_schemas)
            except LLMError:
                raise AgentError("Agent stopped because model communication failed.") from None

            if not response.tool_calls:
                if response.text.strip():
                    if state.workspace_changed and state.changes_since_verification:
                        history.extend(response.continuation_items)
                        history.append(
                            {
                                "role": "system",
                                "content": _VERIFICATION_REQUIRED_OBSERVATION,
                            }
                        )
                        continue
                    return response.text
                raise AgentError("Model response did not contain a final answer or tool call.")

            if not response.continuation_items:
                raise AgentError("Model tool-call response lacked continuation history.")
            history.extend(response.continuation_items)

            tool_count = len(response.tool_calls)
            for tool_index, tool_call in enumerate(response.tool_calls, start=1):
                fingerprint = _tool_call_fingerprint(tool_call)
                if fingerprint == previous_fingerprint:
                    consecutive_repeat_count += 1
                else:
                    previous_fingerprint = fingerprint
                    consecutive_repeat_count = 1

                if consecutive_repeat_count >= _REPEAT_THRESHOLD:
                    result = ToolResult(False, error=_REPEATED_ACTION_ERROR)
                else:
                    result = _dispatch_tool_call(
                        tool_call,
                        self._tool_schemas,
                        self._tool_handlers,
                    )
                _update_task_state(state, tool_call, result)
                if self._trace is not None:
                    self._trace(
                        _format_trace(
                            step,
                            self._max_steps,
                            tool_call,
                            result,
                            tool_index,
                            tool_count,
                        )
                    )
                history.append(_tool_output_item(tool_call.call_id, result))

        raise AgentError(f"Agent exceeded the maximum of {self._max_steps} steps.")


def _load_project_instructions() -> tuple[str | None, str | None]:
    """Load bounded UTF-8 instructions only from the workspace root."""
    try:
        root = config.WORKSPACE_ROOT.resolve()
        requested = root / "AGENTS.md"
        if not requested.exists():
            return None, None
        instruction_file = requested.resolve(strict=True)
        instruction_file.relative_to(root)
        if not instruction_file.is_file():
            raise OSError("AGENTS.md is not a regular file")
        with instruction_file.open("r", encoding="utf-8") as stream:
            content = stream.read(_MAX_PROJECT_INSTRUCTIONS + 1)
    except (OSError, UnicodeDecodeError, ValueError):
        return None, "Project instructions: could not load AGENTS.md"

    if len(content) > _MAX_PROJECT_INSTRUCTIONS:
        available = _MAX_PROJECT_INSTRUCTIONS - len(_PROJECT_INSTRUCTIONS_TRUNCATION)
        content = content[:available] + _PROJECT_INSTRUCTIONS_TRUNCATION
    return content, "Project instructions: loaded AGENTS.md"


def _tool_call_fingerprint(tool_call: ToolCall) -> str:
    """Create a deterministic identity for consecutive-action detection."""
    normalized_arguments = json.dumps(
        tool_call.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{tool_call.name}|{normalized_arguments}"


def _update_task_state(
    state: TaskState, tool_call: ToolCall, result: ToolResult
) -> None:
    """Record only successful, controller-observable execution facts."""
    if not result.success:
        return
    if tool_call.name in {"write_file", "edit_file"}:
        state.workspace_changed = True
        state.changes_since_verification = True
        return
    if tool_call.name != "run_command" or not state.changes_since_verification:
        return

    command = tool_call.arguments.get("command")
    if not isinstance(command, str):
        return
    state.changes_since_verification = False
    state.last_verification_command = command
    state.verification_evidence = _concise_verification_evidence(result.output)


def _concise_verification_evidence(output: str) -> str:
    """Keep bounded command evidence without turning TaskState into history."""
    if len(output) <= _VERIFICATION_EVIDENCE_LIMIT:
        return output
    marker = "\n... [verification evidence truncated]"
    available = _VERIFICATION_EVIDENCE_LIMIT - len(marker)
    return output[:available] + marker


def _dispatch_tool_call(
    tool_call: ToolCall,
    tool_schemas: Sequence[dict[str, object]],
    tool_handlers: Mapping[str, Callable[..., ToolResult]],
) -> ToolResult:
    """Validate and execute one requested local tool without raising tool errors."""
    schema = _find_tool_schema(tool_call.name, tool_schemas)
    handler = tool_handlers.get(tool_call.name)
    if schema is None or handler is None:
        return ToolResult(False, error=f"Unknown tool: {tool_call.name}")

    validation_error = _validate_tool_arguments(tool_call.arguments, schema)
    if validation_error is not None:
        return ToolResult(False, error=validation_error)

    try:
        result = handler(**tool_call.arguments)
    except Exception:
        return ToolResult(False, error=f"Tool {tool_call.name} failed unexpectedly.")
    if not isinstance(result, ToolResult):
        return ToolResult(False, error=f"Tool {tool_call.name} returned an invalid result.")
    return result


def _find_tool_schema(
    name: str, tool_schemas: Sequence[dict[str, object]]
) -> dict[str, object] | None:
    """Find a handwritten schema by its exact model-visible name."""
    for schema in tool_schemas:
        if schema.get("name") == name:
            return schema
    return None


def _validate_tool_arguments(
    arguments: dict[str, object], schema: dict[str, object]
) -> str | None:
    """Perform shallow required, extra, and primitive-type validation."""
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return "Tool schema is malformed."
    properties = parameters.get("properties")
    required = parameters.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return "Tool schema is malformed."

    missing = [name for name in required if name not in arguments]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"

    extra = [name for name in arguments if name not in properties]
    if extra:
        return f"Unsupported argument(s): {', '.join(extra)}"

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            return "Tool schema is malformed."
        expected_type = property_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            return f"Argument {name} must be a string."
    return None


def _tool_output_item(call_id: str, result: ToolResult) -> InputItem:
    """Build one Responses API function_call_output history item."""
    serialized_result = json.dumps(
        {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": serialized_result,
    }


def _format_trace(
    step: int,
    max_steps: int,
    tool_call: ToolCall,
    result: ToolResult,
    tool_index: int = 1,
    tool_count: int = 1,
) -> str:
    """Format one observable tool action without provider or reasoning details."""
    safe_arguments: dict[str, object] = {}
    for name, value in tool_call.arguments.items():
        if name == "content" and isinstance(value, str):
            safe_arguments[name] = f"[text content omitted: {len(value)} characters]"
        elif isinstance(value, str):
            safe_arguments[name] = _redact_text(value)
        else:
            safe_arguments[name] = value

    step_heading = f"[Step {step}/{max_steps}]"
    if tool_count > 1:
        step_heading = f"[Step {step}/{max_steps} - Tool {tool_index}/{tool_count}]"

    lines = [
        step_heading,
        "",
        f"Tool: {tool_call.name}",
        "Arguments:",
        json.dumps(safe_arguments, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Result:",
        f"success: {str(result.success).lower()}",
    ]
    if result.output:
        lines.extend(("output:", _redact_text(result.output)))
    if result.error:
        lines.extend(("error:", _redact_text(result.error)))
    return "\n".join(lines)


def _redact_text(text: str) -> str:
    """Hide common credential assignments and API-key-shaped tokens in trace text."""
    redacted = _SECRET_ASSIGNMENT.sub(r"\1[redacted]", text)
    return _SECRET_TOKEN.sub("[redacted]", redacted)
