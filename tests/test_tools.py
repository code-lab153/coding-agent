"""Tests for the model-independent local tool layer."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

import config
from tools import (
    TOOL_HANDLERS,
    TOOL_SCHEMAS,
    list_files,
    read_file,
    run_command,
    write_file,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give every test a new workspace unrelated to the real project files."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", root)
    return root


def python_command(code: str) -> str:
    """Build a command string that invokes this test environment's Python."""
    arguments = [sys.executable, "-c", code]
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def test_list_files_existing_directory(workspace: Path) -> None:
    (workspace / "zeta.txt").write_text("z", encoding="utf-8")
    (workspace / "alpha").mkdir()

    result = list_files()

    assert result.success
    assert result.output.splitlines() == ["[D] alpha", "[F] zeta.txt"]


def test_list_files_missing_path(workspace: Path) -> None:
    result = list_files("missing")

    assert not result.success
    assert "does not exist" in (result.error or "")


def test_list_files_rejects_file(workspace: Path) -> None:
    (workspace / "item.txt").write_text("content", encoding="utf-8")

    result = list_files("item.txt")

    assert not result.success
    assert "not a directory" in (result.error or "")


def test_list_files_rejects_outside_workspace(workspace: Path) -> None:
    result = list_files("../")

    assert not result.success
    assert "outside the workspace" in (result.error or "")


def test_read_file_reads_utf8(workspace: Path) -> None:
    (workspace / "hello.txt").write_text("hello, world", encoding="utf-8")

    result = read_file("hello.txt")

    assert result.success
    assert result.output == "hello, world"


def test_read_file_missing_file(workspace: Path) -> None:
    result = read_file("missing.txt")

    assert not result.success
    assert "does not exist" in (result.error or "")


def test_read_file_rejects_directory(workspace: Path) -> None:
    result = read_file(".")

    assert not result.success
    assert "not a file" in (result.error or "")


def test_read_file_rejects_path_traversal(workspace: Path) -> None:
    result = read_file("../../outside.txt")

    assert not result.success
    assert "outside the workspace" in (result.error or "")


def test_read_file_rejects_absolute_path(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    result = read_file(str(outside.resolve()))

    assert not result.success
    assert "Absolute paths are not allowed" in (result.error or "")


def test_read_file_rejects_symlink_escape(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this system.")

    result = read_file("link.txt")

    assert not result.success
    assert "outside the workspace" in (result.error or "")


def test_read_file_rejects_non_utf8_text(workspace: Path) -> None:
    (workspace / "binary.dat").write_bytes(b"\xff\xfe\xfd")

    result = read_file("binary.dat")

    assert not result.success
    assert "not valid UTF-8" in (result.error or "")


def test_read_file_truncates_large_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT", 50)
    (workspace / "large.txt").write_text("x" * 500, encoding="utf-8")

    result = read_file("large.txt")

    assert result.success
    assert len(result.output) <= 50
    assert "truncated" in result.output


def test_write_file_creates_file(workspace: Path) -> None:
    result = write_file("new.txt", "new content\n")

    assert result.success
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "new content\n"


def test_write_file_overwrites_file(workspace: Path) -> None:
    (workspace / "item.txt").write_text("old", encoding="utf-8")

    result = write_file("item.txt", "new")

    assert result.success
    assert (workspace / "item.txt").read_text(encoding="utf-8") == "new"


def test_write_file_creates_parent_directories(workspace: Path) -> None:
    result = write_file("nested/deep/item.txt", "content")

    assert result.success
    assert (workspace / "nested/deep/item.txt").is_file()


def test_write_file_rejects_path_traversal(workspace: Path) -> None:
    result = write_file("../outside.txt", "content")

    assert not result.success
    assert "outside the workspace" in (result.error or "")


def test_write_file_returns_unified_diff(workspace: Path) -> None:
    (workspace / "item.txt").write_text("old line\n", encoding="utf-8")

    result = write_file("item.txt", "new line\n")

    assert result.success
    assert "--- a/item.txt" in result.output
    assert "+++ b/item.txt" in result.output
    assert "-old line" in result.output
    assert "+new line" in result.output


def test_run_command_succeeds(workspace: Path) -> None:
    result = run_command(python_command("pass"))

    assert result.success
    assert "returncode: 0" in result.output


def test_run_command_captures_stdout(workspace: Path) -> None:
    result = run_command(python_command("print('hello')"))

    assert result.success
    assert "hello" in result.output


def test_run_command_captures_stderr(workspace: Path) -> None:
    result = run_command(python_command("import sys; sys.stderr.write('problem')"))

    assert result.success
    assert "problem" in result.output


def test_run_command_preserves_nonzero_exit(workspace: Path) -> None:
    result = run_command(python_command("raise SystemExit(7)"))

    assert not result.success
    assert "returncode: 7" in result.output
    assert "return code 7" in (result.error or "")


def test_run_command_times_out(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "COMMAND_TIMEOUT", 0.05)

    result = run_command(python_command("import time; time.sleep(1)"))

    assert not result.success
    assert "timed out" in (result.error or "")


def test_run_command_uses_workspace_as_cwd(workspace: Path) -> None:
    result = run_command(python_command("import os; print(os.getcwd())"))

    assert result.success
    assert str(workspace) in result.output


def test_run_command_truncates_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT", 80)

    result = run_command(python_command("print('x' * 500)"))

    assert result.success
    assert len(result.output) <= 80
    assert "truncated" in result.output


def test_tool_schemas_cover_all_local_tools() -> None:
    names = {schema["name"] for schema in TOOL_SCHEMAS}

    assert names == {"list_files", "read_file", "write_file", "run_command"}
    assert set(TOOL_HANDLERS) == names
    assert all(callable(handler) for handler in TOOL_HANDLERS.values())


def test_tool_schema_required_arguments_are_accurate() -> None:
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    assert schemas["list_files"]["parameters"]["required"] == []
    assert schemas["read_file"]["parameters"]["required"] == ["path"]
    assert schemas["write_file"]["parameters"]["required"] == ["path", "content"]
    assert schemas["run_command"]["parameters"]["required"] == ["command"]


def test_tool_schemas_are_explicit_function_definitions() -> None:
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert isinstance(schema["name"], str)
        assert isinstance(schema["description"], str)
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["additionalProperties"] is False

    assert TOOL_SCHEMAS[0]["strict"] is False
    assert all(schema["strict"] is True for schema in TOOL_SCHEMAS[1:])


def test_tool_schemas_do_not_expose_local_identity_or_absolute_paths() -> None:
    serialized = json.dumps(TOOL_SCHEMAS).lower()

    assert ":\\\\" not in serialized
    assert ":/" not in serialized
    assert "workspace_root" not in serialized
    assert "users/" not in serialized
