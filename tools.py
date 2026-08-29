"""Small, model-independent tools that operate on the local workspace."""

from collections.abc import Callable
from dataclasses import dataclass
from difflib import unified_diff
import locale
from pathlib import Path
import subprocess

import config


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one local tool execution."""

    success: bool
    output: str = ""
    error: str | None = None


class WorkspacePathError(ValueError):
    """Raised when a requested path would leave the configured workspace."""


def _truncate(text: str) -> str:
    """Bound text to the configured output limit and mark truncation."""
    limit = config.MAX_TOOL_OUTPUT
    if len(text) <= limit:
        return text

    marker = "\n... [output truncated]"
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _resolve_workspace_path(path: str) -> Path:
    """Resolve a relative path and reject anything outside the workspace."""
    if not isinstance(path, str):
        raise WorkspacePathError("Path must be a string relative to the workspace.")

    requested = Path(path)
    if requested.is_absolute():
        raise WorkspacePathError("Absolute paths are not allowed; use a workspace-relative path.")

    root = config.WORKSPACE_ROOT.resolve()
    resolved = (root / requested).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("Path is outside the workspace.") from exc
    return resolved


def list_files(path: str = ".") -> ToolResult:
    """List the immediate children of a workspace directory."""
    try:
        directory = _resolve_workspace_path(path)
        if not directory.exists():
            return ToolResult(False, error=f"Directory does not exist: {path}")
        if not directory.is_dir():
            return ToolResult(False, error=f"Path is not a directory: {path}")

        entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        if not entries:
            return ToolResult(True, "(empty directory)")

        lines: list[str] = []
        for entry in entries:
            if entry.is_symlink():
                kind = "[L]"
            elif entry.is_dir():
                kind = "[D]"
            else:
                kind = "[F]"
            display_name = entry.name.replace("\n", "\\n")
            lines.append(f"{kind} {display_name}")
        return ToolResult(True, _truncate("\n".join(lines)))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except OSError as exc:
        return ToolResult(False, error=f"Could not list directory: {exc}")


def read_file(path: str) -> ToolResult:
    """Read a UTF-8 text file inside the workspace."""
    try:
        file_path = _resolve_workspace_path(path)
        if not file_path.exists():
            return ToolResult(False, error=f"File does not exist: {path}")
        if not file_path.is_file():
            return ToolResult(False, error=f"Path is not a file: {path}")

        content = file_path.read_text(encoding="utf-8")
        return ToolResult(True, _truncate(content))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except UnicodeDecodeError:
        return ToolResult(False, error=f"File is not valid UTF-8 text: {path}")
    except OSError as exc:
        return ToolResult(False, error=f"Could not read file: {exc}")


def write_file(path: str, content: str) -> ToolResult:
    """Create or replace a UTF-8 text file and report a unified diff."""
    if not isinstance(content, str):
        return ToolResult(False, error="Content must be a string.")

    try:
        file_path = _resolve_workspace_path(path)
        if file_path.exists() and file_path.is_dir():
            return ToolResult(False, error=f"Path is a directory: {path}")

        old_content: str | None = ""
        if file_path.exists():
            try:
                old_content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                old_content = None

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8", newline="")

        relative_name = file_path.relative_to(config.WORKSPACE_ROOT.resolve()).as_posix()
        summary = f"Wrote {len(content)} characters to {relative_name}."
        if old_content is None:
            summary += "\nUnified diff unavailable because the previous content was not UTF-8."
        else:
            diff = "\n".join(
                unified_diff(
                    old_content.splitlines(),
                    content.splitlines(),
                    fromfile=f"a/{relative_name}",
                    tofile=f"b/{relative_name}",
                    lineterm="",
                )
            )
            summary += f"\n\n{diff}" if diff else "\nNo content changes."
        return ToolResult(True, _truncate(summary))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except OSError as exc:
        return ToolResult(False, error=f"Could not write file: {exc}")


def _as_text(value: str | bytes | None) -> str:
    """Normalize subprocess output, including partial timeout output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(locale.getpreferredencoding(False), errors="replace")
    return value


def _format_command_output(returncode: int | str, stdout: str, stderr: str) -> str:
    """Format captured process information for a future model to inspect."""
    text = (
        f"returncode: {returncode}\n"
        f"stdout:\n{stdout or '(empty)'}\n"
        f"stderr:\n{stderr or '(empty)'}"
    )
    return _truncate(text)


def run_command(command: str) -> ToolResult:
    """Run a shell command locally with the workspace as its working directory.

    Setting ``cwd`` controls the starting directory but is not a security
    sandbox: a shell command can still access other locations allowed by the
    operating system.
    """
    if not isinstance(command, str) or not command.strip():
        return ToolResult(False, error="Command must be a non-empty string.")

    workspace = config.WORKSPACE_ROOT.resolve()
    if not workspace.exists() or not workspace.is_dir():
        return ToolResult(False, error="Configured workspace directory does not exist.")

    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            timeout=config.COMMAND_TIMEOUT,
        )
        stdout = _as_text(completed.stdout)
        stderr = _as_text(completed.stderr)
        output = _format_command_output(completed.returncode, stdout, stderr)
        if completed.returncode == 0:
            return ToolResult(True, output)
        return ToolResult(
            False,
            output,
            error=f"Command exited with return code {completed.returncode}.",
        )
    except subprocess.TimeoutExpired as exc:
        output = _format_command_output(
            "timeout", _as_text(exc.stdout), _as_text(exc.stderr)
        )
        return ToolResult(
            False,
            output,
            error=f"Command timed out after {config.COMMAND_TIMEOUT} seconds.",
        )
    except OSError as exc:
        return ToolResult(False, error=f"Could not start command: {exc}")


TOOL_SCHEMAS: tuple[dict[str, object], ...] = (
    {
        "type": "function",
        "name": "list_files",
        "description": "List files and directories in a workspace-relative directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the workspace; omit for the root.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Create or replace a UTF-8 text file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete text content to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command locally from the workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
)


TOOL_HANDLERS: dict[str, Callable[..., ToolResult]] = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}
