"""Small, model-independent tools that operate on the local workspace."""

from collections.abc import Callable
from dataclasses import dataclass
from difflib import unified_diff
import locale
import os
from pathlib import Path
import shlex
import subprocess
import sys

import config


_SKIPPED_SEARCH_DIRECTORIES = frozenset(
    {"__pycache__", ".pytest_cache", ".git", ".venv"}
)
_BINARY_SEARCH_SUFFIXES = frozenset(
    {
        ".bin",
        ".bmp",
        ".dll",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mp3",
        ".mp4",
        ".mov",
        ".pdf",
        ".png",
        ".pyc",
        ".pyd",
        ".so",
        ".tar",
        ".ttf",
        ".woff",
        ".woff2",
        ".zip",
    }
)
# Bound location lists before applying the shared character-output limit.
_MAX_SEARCH_MATCHES = 200


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


def _unified_text_diff(old_content: str, new_content: str, relative_name: str) -> str:
    """Return the deterministic unified diff used by file-writing tools."""
    return "\n".join(
        unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{relative_name}",
            tofile=f"b/{relative_name}",
            lineterm="",
        )
    )


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


def search_text(query: str, path: str = ".") -> ToolResult:
    """Find literal text in one UTF-8 file or a directory tree."""
    if not isinstance(query, str) or not query:
        return ToolResult(False, error="Query must be a non-empty string.")

    try:
        target = _resolve_workspace_path(path)
        if not target.exists():
            return ToolResult(False, error=f"Path does not exist: {path}")
        if not target.is_file() and not target.is_dir():
            return ToolResult(False, error=f"Path is not a file or directory: {path}")

        workspace = config.WORKSPACE_ROOT.resolve()
        direct_file = target.is_file()
        if direct_file:
            candidates = [target]
        else:
            candidates: list[Path] = []
            for current_root, directory_names, file_names in os.walk(target):
                current_path = Path(current_root)
                directory_names[:] = sorted(
                    (
                        name
                        for name in directory_names
                        if name.casefold() not in _SKIPPED_SEARCH_DIRECTORIES
                        and not (current_path / name).is_symlink()
                    ),
                    key=lambda name: (name.casefold(), name),
                )
                for file_name in sorted(
                    file_names, key=lambda name: (name.casefold(), name)
                ):
                    candidate = current_path / file_name
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    if candidate.suffix.casefold() in _BINARY_SEARCH_SUFFIXES:
                        continue
                    candidates.append(candidate)
            candidates.sort(
                key=lambda candidate: candidate.relative_to(workspace).as_posix()
            )

        matches: list[str] = []
        match_limit_reached = False
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
                candidate.relative_to(workspace)
            except (OSError, ValueError):
                continue
            if candidate.suffix.casefold() in _BINARY_SEARCH_SUFFIXES:
                if direct_file:
                    return ToolResult(False, error=f"File type is not searched as text: {path}")
                continue
            try:
                content = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                if direct_file:
                    return ToolResult(False, error=f"File is not valid UTF-8 text: {path}")
                continue
            except OSError as exc:
                if direct_file:
                    return ToolResult(False, error=f"Could not read file: {exc}")
                continue

            relative_name = candidate.relative_to(workspace).as_posix()
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query not in line:
                    continue
                matches.append(f"{relative_name}:{line_number}:{line}")
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    match_limit_reached = True
                    break
            if match_limit_reached:
                break

        if not matches:
            return ToolResult(True, "No matches found.")
        if match_limit_reached:
            matches.append(
                f"... [match limit reached after {_MAX_SEARCH_MATCHES} matches]"
            )
        return ToolResult(True, _truncate("\n".join(matches)))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except OSError as exc:
        return ToolResult(False, error=f"Could not search path: {exc}")


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
            diff = _unified_text_diff(old_content, content, relative_name)
            summary += f"\n\n{diff}" if diff else "\nNo content changes."
        return ToolResult(True, _truncate(summary))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except OSError as exc:
        return ToolResult(False, error=f"Could not write file: {exc}")


def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    """Replace one uniquely matching text block in an existing UTF-8 file."""
    if not isinstance(old_text, str) or not old_text:
        return ToolResult(False, error="old_text must be a non-empty string.")
    if not isinstance(new_text, str):
        return ToolResult(False, error="new_text must be a string.")

    try:
        file_path = _resolve_workspace_path(path)
        if not file_path.exists():
            return ToolResult(False, error=f"File does not exist: {path}")
        if not file_path.is_file():
            return ToolResult(False, error=f"Path is not a file: {path}")

        old_content = file_path.read_text(encoding="utf-8")
        occurrence_count = old_content.count(old_text)
        if occurrence_count == 0:
            return ToolResult(False, error="The requested text was not found.")
        if occurrence_count > 1:
            return ToolResult(
                False,
                error=(
                    "The requested text is ambiguous because it occurs "
                    f"{occurrence_count} times."
                ),
            )

        new_content = old_content.replace(old_text, new_text, 1)
        file_path.write_text(new_content, encoding="utf-8", newline="")
        relative_name = file_path.relative_to(
            config.WORKSPACE_ROOT.resolve()
        ).as_posix()
        diff = _unified_text_diff(old_content, new_content, relative_name)
        summary = f"Edited {relative_name}."
        summary += f"\n\n{diff}" if diff else "\nNo content changes."
        return ToolResult(True, _truncate(summary))
    except WorkspacePathError as exc:
        return ToolResult(False, error=str(exc))
    except UnicodeDecodeError:
        return ToolResult(False, error=f"File is not valid UTF-8 text: {path}")
    except OSError as exc:
        return ToolResult(False, error=f"Could not edit file: {exc}")


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


def _subprocess_environment() -> dict[str, str]:
    """Copy the process environment and prioritize the launching Python directory."""
    environment = os.environ.copy()
    python_directory = str(Path(sys.executable).parent)
    existing_path = environment.get("PATH", "")
    environment["PATH"] = (
        python_directory + os.pathsep + existing_path
        if existing_path
        else python_directory
    )
    return environment


def _unquoted_shell_operator(command: str) -> str | None:
    """Find shell composition outside quotes without pretending to fully parse it."""
    quote: str | None = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "\\" and quote == '"':
                index += 1
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in {"&", "|", ";", ">", "<", "\n", "\r"}:
            return repr(character)
        index += 1
    if quote is not None:
        return "an unclosed quote"
    return None


def _command_tokens(command: str) -> list[str] | None:
    """Split a simple command only far enough for explicit policy decisions."""
    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return None
    return [token.strip('"\'') for token in tokens]


def _executable_name(token: str) -> str:
    """Normalize a command executable name across Windows and POSIX paths."""
    name = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    for suffix in (".exe", ".cmd", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _git_subcommand(arguments: list[str]) -> str | None:
    """Locate a Git subcommand after the common global options we support."""
    index = 0
    options_with_values = {"-c", "-C", "--git-dir", "--work-tree", "--namespace"}
    while index < len(arguments):
        argument = arguments[index]
        if argument in options_with_values:
            index += 2
            continue
        if argument.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument.casefold()
    return None


def _command_policy_denial(command: str) -> str | None:
    """Return a reason when a simple command is clearly unsafe or out of scope."""
    operator = _unquoted_shell_operator(command)
    if operator is not None:
        return f"complex shell syntax ({operator}) is not allowed"

    tokens = _command_tokens(command)
    if not tokens:
        return "the command could not be parsed safely"

    executable = _executable_name(tokens[0])
    arguments = tokens[1:]
    lowered = [argument.casefold() for argument in arguments]

    if executable == "git":
        subcommand = _git_subcommand(arguments)
        if subcommand in {"push", "reset", "clean", "clone", "fetch", "pull"}:
            return f"git {subcommand} can change external or repository state"

    if executable in {"rm", "rmdir", "unlink", "shred", "del", "erase", "rd", "remove-item"}:
        return "filesystem deletion commands are not allowed"
    if executable in {"shutdown", "reboot", "poweroff", "halt"}:
        return "system power commands are not allowed"
    if executable in {"format", "diskpart", "fdisk", "parted"} or executable.startswith("mkfs"):
        return "disk formatting or partition commands are not allowed"
    if executable in {"curl", "wget", "invoke-webrequest", "invoke-restmethod", "iwr", "irm"}:
        return "network download commands are not allowed"
    if executable in {"powershell", "pwsh", "cmd", "sh", "bash", "zsh"}:
        return "nested shell commands are not allowed"

    if executable in {"pip", "pip3"} and lowered[:1] in (["install"], ["uninstall"]):
        return "package installation commands are not allowed"
    if executable in {"python", "python3", "py"} and len(lowered) >= 3:
        if lowered[0:2] == ["-m", "pip"] and lowered[2] in {"install", "uninstall"}:
            return "package installation commands are not allowed"
    package_actions = {
        "npm": {"install", "i", "ci", "uninstall"},
        "yarn": {"add", "install", "remove"},
        "pnpm": {"add", "install", "remove"},
        "cargo": {"install"},
        "gem": {"install", "uninstall"},
        "apt": {"install", "remove", "upgrade"},
        "apt-get": {"install", "remove", "upgrade"},
        "dnf": {"install", "remove", "upgrade"},
        "yum": {"install", "remove", "update"},
        "brew": {"install", "uninstall", "upgrade"},
        "winget": {"install", "uninstall", "upgrade"},
        "choco": {"install", "uninstall", "upgrade"},
        "conda": {"install", "remove", "update"},
    }
    if lowered and lowered[0] in package_actions.get(executable, set()):
        return "package installation commands are not allowed"
    return None


def run_command(command: str) -> ToolResult:
    """Run a shell command locally with the workspace as its working directory.

    Setting ``cwd`` controls the starting directory but is not a security
    sandbox: a shell command can still access other locations allowed by the
    operating system.
    """
    if not isinstance(command, str) or not command.strip():
        return ToolResult(False, error="Command must be a non-empty string.")

    policy_denial = _command_policy_denial(command)
    if policy_denial is not None:
        return ToolResult(False, error=f"Command denied by safety policy: {policy_denial}.")

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
            env=_subprocess_environment(),
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
        "name": "search_text",
        "description": "Find literal text in a workspace file or directory tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal text to find.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path relative to the workspace; omit for the root.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
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
        "name": "edit_file",
        "description": "Replace one uniquely matching text block in an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact existing text that must occur once.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
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
    "search_text": search_text,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
}
