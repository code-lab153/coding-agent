"""Small synchronous Agent loop and local tool-dispatch boundary."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re

import config
from config import MAX_STEPS
from context_manager import ContextManager
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

    original_goal: str  #初始的任务
    workspace_changed: bool = False #工作区是否被修改过
    changes_since_verification: bool = False    #表示是不是有未验证的修改
    last_verification_command: str | None = None    #最后一次成功的验证命令
    verification_evidence: str | None = None    #记录验证命令的输出（一小部分，最多500字符左右），方便后续分析和调试


class Agent:
    """Coordinate model decisions, local tool actions, and conversation history."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_schemas: Sequence[dict[str, object]] = TOOL_SCHEMAS,   #给大模型看的工具说明
        tool_handlers: Mapping[str, Callable[..., ToolResult]] = TOOL_HANDLERS, #可以真正执行的python函数（工具）
        max_steps: int = MAX_STEPS, #最多请求模型的轮数
        trace: Callable[[str], None] | None = None, #运行过程的输出函数，每次工具执行都会打印到终端
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
        self.canonical_history: list[InputItem] = []    #保存完整的运行历史，包括模型的输出和工具的输出，方便后续分析和调试

    def run(self, task: str) -> str:
        """Run the decide-act-observe loop until text completion or a fatal error."""
        #检查任务是否合法
        if not isinstance(task, str) or not task.strip():
            raise AgentError("Task must be a non-empty string.")

        #初始化任务状态，原始目标即为用户输入的任务
        state = TaskState(original_goal=task)
        self.task_state = state
        """
        加载项目的规则，此函数会查找：workspace/AGENTS.md，若存在则读取里面的项目规则（包含project_instructions, instruction_trace两类）
        eg：
        不要修改验收测试
        不要安装第三方依赖
        使用 tkinter
        完成前运行测试
        """
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
        self.canonical_history = history
        context_manager = ContextManager(static_item_count=len(history))
        previous_fingerprint: str | None = None
        consecutive_repeat_count = 0

        #进入agent循环，每次循环会向大模型发送消息，获取大模型的输出，判断是否有工具调用，如果有则执行工具调用，并将工具调用的结果加入历史中
        #最多20轮
        for step in range(1, self._max_steps + 1):
            try:
                #每一轮先整理上下文，包括历史消息和任务状态，构建模型输入
                model_input = context_manager.build_context(history, state)
                response = self._llm_client.send(model_input, tools=self._tool_schemas) #调用模型并得到返回
            except LLMError:
                raise AgentError("Agent stopped because model communication failed.") from None

            #首先检查模型的输出中是否有工具调用，如果没有工具调用，则说明模型已经给出了最终答案，直接返回即可
            if not response.tool_calls:
                #顺便判断模型的输出是否为空，如果为空，则说明模型没有给出最终答案，也没有工具调用，说明模型的输出不合法，抛出异常
                if response.text.strip():
                    #若工作区已经被修改，并且最新修改还没有被验证，则不允许结束
                    if state.workspace_changed and state.changes_since_verification:
                        history.extend(response.continuation_items)
                        """
                        并且还会加入一条系统提示：eg：
                            工作区已经发生修改，
                            但最新修改后没有成功运行验证命令。
                            请先运行适当的验证命令。
                        """
                        history.append(
                            {
                                "role": "system",
                                "content": _VERIFICATION_REQUIRED_OBSERVATION,
                            }
                        )
                        context_manager.record_completed_step(len(history))
                        #进入下一轮
                        continue
                    history.extend(response.continuation_items)
                    return response.text
                raise AgentError("Model response did not contain a final answer or tool call.")

            #如果返回没有总结好的历史信息，则说明模型的输出不合法，抛出异常，否则加入历史中
            if not response.continuation_items:
                raise AgentError("Model tool-call response lacked continuation history.")
            history.extend(response.continuation_items)

            tool_count = len(response.tool_calls)
            #遍历模型返回的工具调用列表，执行每一个工具调用，并将结果加入历史中
            #但在本项目中，每次模型调用只会返回一个工具调用，所以这里的循环实际上只会执行一次
            for tool_index, tool_call in enumerate(response.tool_calls, start=1):
                #用于检测连续重复的工具调用，若连续重复超过3次，则不再执行工具调用，而是直接返回错误
                #用工具名和参数来生成一个指纹，若连续3次指纹相同，则说明连续重复调用了同一个工具
                fingerprint = _tool_call_fingerprint(tool_call)
                if fingerprint == previous_fingerprint:
                    consecutive_repeat_count += 1
                else:
                    previous_fingerprint = fingerprint
                    consecutive_repeat_count = 1

                #若连续重复调用超过3次，则不再执行工具调用，而是直接返回错误
                #否则执行工具调用，并将结果加入历史中
                if consecutive_repeat_count >= _REPEAT_THRESHOLD:
                    result = ToolResult(False, error=_REPEATED_ACTION_ERROR)
                #若没超过次数，则执行工具调用，并将结果加入历史中
                else:
                    result = _dispatch_tool_call(
                        tool_call,
                        self._tool_schemas,
                        self._tool_handlers,
                    )
                #更新taskstate
                """
                一共有六种工具的调用，但是只有三种工具的调用会修改工作区，分别是：write_file, edit_file, run_command
                其中，write_file和edit_file会直接修改工作区，run_command则是执行命令，可能会修改工作区，也可能不会修改工作区，所以需要判断一下
                （1）如果执行write_file或edit_file，则将workspace_changed和changes_since_verification都置为True
                （2）如果执行run_command，则需要判断
                """
                _update_task_state(state, tool_call, result)
                #如果传入了trace函数，则将工具调用的结果打印出来，方便调试
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
                #将工具的结果加入到历史中，方便后续的模型调用使用
                history.append(_tool_output_item(tool_call.call_id, result))
            context_manager.record_completed_step(len(history))

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
    #如果失败不更新状态
    if not result.success:
        return
    if tool_call.name in {"write_file", "edit_file"}:
        state.workspace_changed = True  #表示工作区已经被修改过
        state.changes_since_verification = True #表示有未验证的修改
        return
    #如果当前工具调用是run_command，并且当前工作区有未验证的修改，则更新状态，继续向下执行
    if tool_call.name != "run_command" or not state.changes_since_verification:
        return

    #得到执行命令的参数，如果参数中没有command，则不更新状态
    command = tool_call.arguments.get("command")
    if not isinstance(command, str):
        return
    state.changes_since_verification = False    #把未验证变成已验证
    state.last_verification_command = command   #保存验证命令
    state.verification_evidence = _concise_verification_evidence(result.output) #保存验证证据


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
