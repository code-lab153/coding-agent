"""Small synchronous Agent loop and local tool-dispatch boundary."""

from collections.abc import Callable, Mapping, Sequence
import json

from config import MAX_STEPS
from llm import InputItem, LLMClient, LLMError, ToolCall
from prompts import SYSTEM_PROMPT
from tools import TOOL_HANDLERS, TOOL_SCHEMAS, ToolResult


class AgentError(RuntimeError):
    """Raised when the Agent controller cannot safely continue."""


class Agent:
    """Coordinate model decisions, local tool actions, and conversation history."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_schemas: Sequence[dict[str, object]] = TOOL_SCHEMAS,
        tool_handlers: Mapping[str, Callable[..., ToolResult]] = TOOL_HANDLERS,
        max_steps: int = MAX_STEPS,
    ) -> None:
        """Store the explicit collaborators and finite loop budget."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise AgentError("max_steps must be a positive integer.")
        self._llm_client = llm_client
        self._tool_schemas = tuple(tool_schemas)
        self._tool_handlers = dict(tool_handlers)
        self._max_steps = max_steps

    def run(self, task: str) -> str:
        """Run the decide-act-observe loop until text completion or a fatal error."""
        if not isinstance(task, str) or not task.strip():
            raise AgentError("Task must be a non-empty string.")

        history: list[InputItem] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for _step in range(self._max_steps):
            try:
                response = self._llm_client.send(history, tools=self._tool_schemas)
            except LLMError:
                raise AgentError("Agent stopped because model communication failed.") from None

            if not response.tool_calls:
                if response.text.strip():
                    return response.text
                raise AgentError("Model response did not contain a final answer or tool call.")

            if not response.continuation_items:
                raise AgentError("Model tool-call response lacked continuation history.")
            history.extend(response.continuation_items)

            for tool_call in response.tool_calls:
                result = _dispatch_tool_call(
                    tool_call,
                    self._tool_schemas,
                    self._tool_handlers,
                )
                history.append(_tool_output_item(tool_call.call_id, result))

        raise AgentError(f"Agent exceeded the maximum of {self._max_steps} steps.")


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
