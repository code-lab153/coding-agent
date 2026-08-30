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
    edit_file,
    list_files,
    read_file,
    run_command,
    search_text,
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


def test_edit_file_replaces_one_line_and_returns_diff(workspace: Path) -> None:
    target = workspace / "item.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = edit_file("item.py", "value = 1", "value = 2")

    assert result.success
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert "--- a/item.py" in result.output
    assert "+++ b/item.py" in result.output
    assert "-value = 1" in result.output
    assert "+value = 2" in result.output


def test_edit_file_replaces_multiline_text(workspace: Path) -> None:
    target = workspace / "module.py"
    target.write_text(
        "def total():\n    value = 1\n    return value\n",
        encoding="utf-8",
    )

    result = edit_file(
        "module.py",
        "    value = 1\n    return value",
        "    value = 2\n    return value * 2",
    )

    assert result.success
    assert target.read_text(encoding="utf-8") == (
        "def total():\n    value = 2\n    return value * 2\n"
    )


def test_edit_file_preserves_unicode(workspace: Path) -> None:
    target = workspace / "message.txt"
    target.write_text("问候：你好，世界\n", encoding="utf-8")

    result = edit_file("message.txt", "你好，世界", "你好，Agent")

    assert result.success
    assert target.read_text(encoding="utf-8") == "问候：你好，Agent\n"
    assert "+问候：你好，Agent" in result.output


def test_edit_file_fails_when_text_is_not_found(workspace: Path) -> None:
    target = workspace / "item.txt"
    target.write_text("present", encoding="utf-8")

    result = edit_file("item.txt", "missing", "replacement")

    assert not result.success
    assert result.error == "The requested text was not found."
    assert target.read_text(encoding="utf-8") == "present"


def test_edit_file_fails_when_match_is_ambiguous(workspace: Path) -> None:
    target = workspace / "item.txt"
    target.write_text("same\nsame\n", encoding="utf-8")

    result = edit_file("item.txt", "same", "changed")

    assert not result.success
    assert "ambiguous" in (result.error or "")
    assert "2 times" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


def test_edit_file_rejects_empty_old_text(workspace: Path) -> None:
    (workspace / "item.txt").write_text("content", encoding="utf-8")

    result = edit_file("item.txt", "", "replacement")

    assert not result.success
    assert "non-empty" in (result.error or "")


def test_edit_file_rejects_path_traversal_and_preserves_outside_file(
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    result = edit_file("../outside.txt", "outside", "changed")

    assert not result.success
    assert "outside the workspace" in (result.error or "")
    assert outside.read_text(encoding="utf-8") == "outside"


def test_edit_file_rejects_absolute_path(workspace: Path) -> None:
    target = workspace / "item.txt"
    target.write_text("content", encoding="utf-8")

    result = edit_file(str(target.resolve()), "content", "changed")

    assert not result.success
    assert "Absolute paths are not allowed" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "content"


def test_edit_file_rejects_nonexistent_file(workspace: Path) -> None:
    result = edit_file("missing.txt", "old", "new")

    assert not result.success
    assert "File does not exist" in (result.error or "")


def test_edit_file_rejects_non_utf8_file(workspace: Path) -> None:
    (workspace / "binary.dat").write_bytes(b"\xff\xfe\x00")

    result = edit_file("binary.dat", "old", "new")

    assert not result.success
    assert "not valid UTF-8" in (result.error or "")


def test_search_text_finds_literal_in_direct_file_with_line_number(
    workspace: Path,
) -> None:
    (workspace / "source.py").write_text(
        "first line\nnormalize_name(value)\nlast line\n",
        encoding="utf-8",
    )

    result = search_text("normalize_name", "source.py")

    assert result.success
    assert result.output == "source.py:2:normalize_name(value)"


def test_search_text_recurses_with_deterministic_path_order(workspace: Path) -> None:
    (workspace / "b.py").write_text("target b\n", encoding="utf-8")
    (workspace / "a.py").write_text("target a\n", encoding="utf-8")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "c.py").write_text("target c\n", encoding="utf-8")

    result = search_text("target")

    assert result.success
    assert result.output.splitlines() == [
        "a.py:1:target a",
        "b.py:1:target b",
        "nested/c.py:1:target c",
    ]


def test_search_text_preserves_unicode(workspace: Path) -> None:
    (workspace / "message.txt").write_text(
        "第一行\n你好，世界\n第三行\n", encoding="utf-8"
    )

    result = search_text("你好")

    assert result.success
    assert result.output == "message.txt:2:你好，世界"


def test_search_text_reports_no_matches_clearly(workspace: Path) -> None:
    (workspace / "item.txt").write_text("ordinary content", encoding="utf-8")

    result = search_text("missing")

    assert result.success
    assert result.output == "No matches found."


def test_search_text_rejects_empty_query(workspace: Path) -> None:
    result = search_text("")

    assert not result.success
    assert "non-empty" in (result.error or "")


def test_search_text_rejects_path_traversal(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("target", encoding="utf-8")

    result = search_text("target", "../outside.txt")

    assert not result.success
    assert "outside the workspace" in (result.error or "")


def test_search_text_rejects_absolute_path(workspace: Path) -> None:
    target = workspace / "item.txt"
    target.write_text("target", encoding="utf-8")

    result = search_text("target", str(target.resolve()))

    assert not result.success
    assert "Absolute paths are not allowed" in (result.error or "")


def test_search_text_handles_non_utf8_direct_file(workspace: Path) -> None:
    (workspace / "binary.dat").write_bytes(b"target\xff\xfe")

    result = search_text("target", "binary.dat")

    assert not result.success
    assert "not valid UTF-8" in (result.error or "")


def test_search_text_skips_generated_and_cache_directories(workspace: Path) -> None:
    (workspace / "visible.py").write_text("find_me visible\n", encoding="utf-8")
    for directory_name in ("__pycache__", ".pytest_cache", ".git", ".venv"):
        directory = workspace / directory_name
        directory.mkdir()
        (directory / "hidden.py").write_text("find_me hidden\n", encoding="utf-8")

    result = search_text("find_me")

    assert result.success
    assert result.output == "visible.py:1:find_me visible"


def test_search_text_skips_binary_files_during_directory_search(
    workspace: Path,
) -> None:
    (workspace / "image.png").write_bytes(b"target\xff\xfe")
    (workspace / "source.py").write_text("target text\n", encoding="utf-8")

    result = search_text("target")

    assert result.success
    assert result.output == "source.py:1:target text"


def test_search_text_output_is_bounded(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT", 90)
    (workspace / "many.txt").write_text(
        "\n".join(f"target line {index}" for index in range(100)),
        encoding="utf-8",
    )

    result = search_text("target")

    assert result.success
    assert len(result.output) <= 90
    assert "truncated" in result.output


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


def test_run_command_prioritizes_launching_python_in_copied_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_path = os.pathsep.join(("existing-first", "existing-second"))
    monkeypatch.setenv("PATH", original_path)
    monkeypatch.setenv("COPY_SENTINEL", "preserved")
    captured: dict[str, object] = {}

    def fake_run(command: str, **options: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(options)
        return subprocess.CompletedProcess(command, 0, "captured stdout", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_command("python -m pytest")

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment is not os.environ
    assert environment["COPY_SENTINEL"] == "preserved"
    assert environment["PATH"].split(os.pathsep) == [
        str(Path(sys.executable).parent),
        "existing-first",
        "existing-second",
    ]
    assert os.environ["PATH"] == original_path
    assert captured["cwd"] == workspace
    assert result.success
    assert "captured stdout" in result.output


def test_run_command_truncates_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT", 80)

    result = run_command(python_command("print('x' * 500)"))

    assert result.success
    assert len(result.output) <= 80
    assert "truncated" in result.output


@pytest.mark.parametrize(
    "command",
    ["python -m pytest", "python script.py", "git status", "git diff"],
)
def test_command_policy_allows_normal_development_commands(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    captured: dict[str, object] = {}

    def fake_run(value: str, **options: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = value
        captured.update(options)
        return subprocess.CompletedProcess(value, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_command(command)

    assert result.success
    assert captured["command"] == command
    assert captured["cwd"] == workspace
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)


def test_command_policy_ignores_operators_inside_quoted_arguments(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_run(command: str, **_options: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_command('python -c "print(\'a|b; c>d\')"')

    assert result.success
    assert called


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest && git status",
        "pytest | more",
        "pytest > result.txt",
        "git status; git diff",
    ],
)
def test_command_policy_denies_complex_shell_composition(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_options: pytest.fail("denied command reached subprocess"),
    )

    result = run_command(command)

    assert not result.success
    assert "denied by safety policy" in (result.error or "")


@pytest.mark.parametrize(
    "command",
    [
        "git push origin main",
        "git reset --hard HEAD~1",
        "git clean -fd",
        "rm -rf build",
        "del /q output.txt",
        "Remove-Item output.txt",
        "shutdown /s",
        "reboot",
        "format C:",
        "diskpart",
        "curl https://example.com/file",
        "wget https://example.com/file",
        "pip install package",
        "python -m pip install package",
        "npm install package",
        "winget install package",
    ],
)
def test_command_policy_denies_dangerous_or_external_commands_before_execution(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_options: pytest.fail("denied command reached subprocess"),
    )

    result = run_command(command)

    assert not result.success
    assert result.output == ""
    assert "Command denied by safety policy" in (result.error or "")


def test_tool_schemas_cover_all_local_tools() -> None:
    names = {schema["name"] for schema in TOOL_SCHEMAS}

    assert names == {
        "list_files",
        "read_file",
        "search_text",
        "write_file",
        "edit_file",
        "run_command",
    }
    assert set(TOOL_HANDLERS) == names
    assert all(callable(handler) for handler in TOOL_HANDLERS.values())


def test_tool_schema_required_arguments_are_accurate() -> None:
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    assert schemas["list_files"]["parameters"]["required"] == []
    assert schemas["read_file"]["parameters"]["required"] == ["path"]
    assert schemas["search_text"]["parameters"]["required"] == ["query"]
    assert schemas["write_file"]["parameters"]["required"] == ["path", "content"]
    assert schemas["edit_file"]["parameters"]["required"] == [
        "path",
        "old_text",
        "new_text",
    ]
    assert schemas["run_command"]["parameters"]["required"] == ["command"]


def test_tool_schemas_are_explicit_function_definitions() -> None:
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert isinstance(schema["name"], str)
        assert isinstance(schema["description"], str)
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["additionalProperties"] is False

    strict_by_name = {schema["name"]: schema["strict"] for schema in TOOL_SCHEMAS}
    assert strict_by_name["list_files"] is False
    assert strict_by_name["search_text"] is False
    assert all(
        strict_by_name[name] is True
        for name in ("read_file", "write_file", "edit_file", "run_command")
    )


def test_tool_schemas_do_not_expose_local_identity_or_absolute_paths() -> None:
    serialized = json.dumps(TOOL_SCHEMAS).lower()

    assert ":\\\\" not in serialized
    assert ":/" not in serialized
    assert "workspace_root" not in serialized
    assert "users/" not in serialized
