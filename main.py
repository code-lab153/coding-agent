"""Small command-line entry point for the coding agent."""

import argparse
from collections.abc import Callable, Sequence
import json
import sys

from agent import Agent, AgentError
from config import ConfigurationError, WORKSPACE_ROOT
from llm import LLMClient, LLMError


_MAX_COMPACT_ARGUMENT_CHARS = 160
_MAX_COMPACT_OUTPUT_CHARS = 900
_MAX_COMPACT_OUTPUT_LINES = 10
_MAX_COMPACT_ERROR_CHARS = 400


def _build_parser() -> argparse.ArgumentParser:
    """Create the intentionally small command-line interface."""
    parser = argparse.ArgumentParser(description="Run the local coding agent.")
    parser.add_argument("task", nargs="?", help="Programming task for the agent.")
    parser.add_argument(
        "--trace",
        choices=("compact", "full"),
        default="compact",
        help="Terminal trace detail (default: compact).",
    )
    return parser


def _trace_printer(mode: str) -> Callable[[str], None]:
    """Choose terminal presentation without changing Agent tool feedback."""
    if mode == "full":
        return print

    def print_compact(message: str) -> None:
        print(_compact_trace_message(message))

    return print_compact


def _compact_trace_message(message: str) -> str:
    """Collapse verbose tool content while preserving actions and outcomes."""
    if not message.startswith("[Step "):
        return message

    request_block, separator, result_block = message.partition("\n\nResult:\n")
    if not separator:
        return message

    request_header, argument_separator, arguments_text = request_block.partition(
        "\nArguments:\n"
    )
    tool_name = _trace_tool_name(request_header)
    if argument_separator:
        arguments_text = _compact_trace_arguments(arguments_text)
        request_block = request_header + argument_separator + arguments_text

    success_line, _, detail_block = result_block.partition("\n")
    output_text, error_text = _split_trace_details(detail_block)
    lines = [request_block, "", "Result:", success_line]
    if output_text is not None:
        lines.extend(("output:", _compact_trace_output(tool_name, output_text)))
    if error_text is not None:
        lines.extend(("error:", _truncate_trace_text(error_text, _MAX_COMPACT_ERROR_CHARS)))
    return "\n".join(lines)


def _trace_tool_name(request_header: str) -> str:
    for line in request_header.splitlines():
        if line.startswith("Tool: "):
            return line.removeprefix("Tool: ").strip()
    return ""


def _compact_trace_arguments(arguments_text: str) -> str:
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError:
        return _truncate_trace_text(arguments_text, _MAX_COMPACT_OUTPUT_CHARS)
    if not isinstance(arguments, dict):
        return arguments_text

    compact: dict[str, object] = {}
    for name, value in arguments.items():
        if isinstance(value, str) and len(value) > _MAX_COMPACT_ARGUMENT_CHARS:
            compact[name] = f"[text omitted from terminal trace: {len(value)} characters]"
        else:
            compact[name] = value
    return json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)


def _split_trace_details(detail_block: str) -> tuple[str | None, str | None]:
    if detail_block.startswith("output:\n"):
        output_and_error = detail_block.removeprefix("output:\n")
        output, error_separator, error = output_and_error.rpartition("\nerror:\n")
        if error_separator:
            return output, error
        return output_and_error, None
    if detail_block.startswith("error:\n"):
        return None, detail_block.removeprefix("error:\n")
    return None, None


def _compact_trace_output(tool_name: str, output: str) -> str:
    if tool_name == "read_file":
        return f"[file content omitted from terminal trace: {len(output)} characters]"

    if tool_name in {"write_file", "edit_file"}:
        first_line = output.splitlines()[0] if output.splitlines() else "(empty)"
        if len(output) > len(first_line):
            return f"{first_line}\n[diff/details omitted: {len(output)} characters total]"
        return first_line

    if len(output) <= _MAX_COMPACT_OUTPUT_CHARS:
        return output

    output_lines = output.splitlines()
    if tool_name == "run_command" and output_lines:
        preview = [output_lines[0], *output_lines[-(_MAX_COMPACT_OUTPUT_LINES - 1) :]]
    else:
        preview = output_lines[:_MAX_COMPACT_OUTPUT_LINES]
    omitted = max(0, len(output_lines) - len(preview))
    return "\n".join(
        [*preview, f"... [{omitted} lines omitted; {len(output)} characters total]"]
    )


def _truncate_trace_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} characters omitted]"


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble the application, run one task, and return a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        #命令行读入task
        task = args.task if args.task is not None else input("Task: ")
        #判断task是否为空
        if not task.strip():
            print("Error: task must not be empty.", file=sys.stderr)
            return 2

        print("Coding Agent")
        print(f"Workspace: {WORKSPACE_ROOT.name}")
        print()

        agent = Agent(LLMClient(), trace=_trace_printer(args.trace))
        final_answer = agent.run(task)
        print("\n[Final]")
        print(final_answer)
        print("\n" + "=" * 50)
        print("[SUCCESS] Agent completed successfully")
        print("=" * 50)
        return 0
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
