"""Offline tests for the small command-line entry point."""

import builtins

import pytest

from agent import AgentError
from config import ConfigurationError
from llm import LLMError
import main as main_module


def install_successful_app(
    monkeypatch: pytest.MonkeyPatch, final_answer: str = "Task complete."
) -> dict[str, object]:
    """Replace production assembly objects and record the CLI handoff."""
    captured: dict[str, object] = {}
    fake_client = object()

    def fake_llm_client() -> object:
        captured["client_created"] = True
        return fake_client

    class FakeAgent:
        def __init__(self, client: object, *, trace: object) -> None:
            captured["client"] = client
            captured["trace"] = trace

        def run(self, task: str) -> str:
            captured["task"] = task
            return final_answer

    monkeypatch.setattr(main_module, "LLMClient", fake_llm_client)
    monkeypatch.setattr(main_module, "Agent", FakeAgent)
    return captured


def test_cli_accepts_task_argument_and_prints_final_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = install_successful_app(monkeypatch, "Fixed and verified.")

    exit_code = main_module.main(["Fix the failing tests."])

    terminal = capsys.readouterr()
    assert exit_code == 0
    assert captured["task"] == "Fix the failing tests."
    assert captured["client"] is not None
    assert callable(captured["trace"])
    assert "Coding Agent" in terminal.out
    assert "Workspace: workspace" in terminal.out
    assert "[Final]\nFixed and verified." in terminal.out
    assert terminal.out.rstrip().endswith(
        "==================================================\n"
        "[SUCCESS] Agent completed successfully\n"
        "=================================================="
    )
    assert terminal.err == ""


def test_cli_reads_interactive_task_when_argument_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_successful_app(monkeypatch)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "Inspect the project"

    monkeypatch.setattr(builtins, "input", fake_input)

    exit_code = main_module.main([])

    assert exit_code == 0
    assert prompts == ["Task: "]
    assert captured["task"] == "Inspect the project"


def test_empty_task_is_rejected_before_client_creation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_client() -> object:
        raise AssertionError("client must not be created for an empty task")

    monkeypatch.setattr(main_module, "LLMClient", forbidden_client)

    exit_code = main_module.main(["   "])

    terminal = capsys.readouterr()
    assert exit_code == 2
    assert "task must not be empty" in terminal.err
    assert terminal.out == ""


def test_configuration_error_is_clean_and_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_client() -> object:
        raise ConfigurationError("MODEL_API_KEY is required.")

    monkeypatch.setattr(main_module, "LLMClient", failing_client)

    exit_code = main_module.main(["Do work"])

    terminal = capsys.readouterr()
    assert exit_code == 1
    assert terminal.err == "Error: MODEL_API_KEY is required.\n"
    assert "Traceback" not in terminal.err


def test_agent_error_is_clean_and_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingAgent:
        def __init__(self, _client: object, *, trace: object) -> None:
            assert callable(trace)

        def run(self, _task: str) -> str:
            raise AgentError("Agent exceeded the maximum of 20 steps.")

    monkeypatch.setattr(main_module, "LLMClient", lambda: object())
    monkeypatch.setattr(main_module, "Agent", FailingAgent)

    exit_code = main_module.main(["Never stop"])

    terminal = capsys.readouterr()
    assert exit_code == 1
    assert "maximum of 20 steps" in terminal.err
    assert "Traceback" not in terminal.err
    assert "[SUCCESS]" not in terminal.out


def test_client_initialization_error_is_clean_and_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_client() -> object:
        raise LLMError("Could not initialize the model API client.")

    monkeypatch.setattr(main_module, "LLMClient", failing_client)

    exit_code = main_module.main(["Do work"])

    terminal = capsys.readouterr()
    assert exit_code == 1
    assert "Could not initialize the model API client" in terminal.err
    assert "Traceback" not in terminal.err


def test_keyboard_interrupt_returns_standard_interrupt_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted_input(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupted_input)

    exit_code = main_module.main([])

    terminal = capsys.readouterr()
    assert exit_code == 130
    assert "Interrupted." in terminal.err
    assert "Traceback" not in terminal.err


def test_configuration_error_does_not_print_secret_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret_value = "test-secret-that-must-stay-hidden"

    def failing_client() -> object:
        raise ConfigurationError("MODEL_API_KEY is invalid.")

    monkeypatch.setattr(main_module, "LLMClient", failing_client)

    main_module.main(["Do work"])

    terminal = capsys.readouterr()
    assert secret_value not in terminal.out
    assert secret_value not in terminal.err


def test_compact_trace_omits_read_file_contents_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    large_content = "private source line\n" * 200
    trace_message = (
        "[Step 2/20]\n\n"
        "Tool: read_file\n"
        'Arguments:\n{\n  "path": "large.py"\n}\n\n'
        "Result:\n"
        "success: true\n"
        f"output:\n{large_content}"
    )

    class EmittingAgent:
        def __init__(self, _client: object, *, trace: object) -> None:
            self.trace = trace

        def run(self, _task: str) -> str:
            assert callable(self.trace)
            self.trace(trace_message)
            return "Done."

    monkeypatch.setattr(main_module, "LLMClient", lambda: object())
    monkeypatch.setattr(main_module, "Agent", EmittingAgent)

    assert main_module.main(["Inspect"]) == 0

    terminal = capsys.readouterr().out
    assert "Tool: read_file" in terminal
    assert '"path": "large.py"' in terminal
    assert "success: true" in terminal
    assert "file content omitted from terminal trace" in terminal
    assert large_content not in terminal


def test_full_trace_mode_preserves_existing_detailed_trace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_message = (
        "[Step 1/20]\n\nTool: read_file\nArguments:\n{}\n\n"
        "Result:\nsuccess: true\noutput:\nFULL-CONTENT-MARKER"
    )

    class EmittingAgent:
        def __init__(self, _client: object, *, trace: object) -> None:
            self.trace = trace

        def run(self, task: str) -> str:
            assert task == "Inspect"
            assert callable(self.trace)
            self.trace(trace_message)
            return "Done."

    monkeypatch.setattr(main_module, "LLMClient", lambda: object())
    monkeypatch.setattr(main_module, "Agent", EmittingAgent)

    assert main_module.main(["--trace", "full", "Inspect"]) == 0

    assert "FULL-CONTENT-MARKER" in capsys.readouterr().out


def test_compact_trace_keeps_short_test_result_visible() -> None:
    message = (
        "[Step 4/20]\n\nTool: run_command\n"
        'Arguments:\n{\n  "command": "python -m pytest -q"\n}\n\n'
        "Result:\nsuccess: true\n"
        "output:\nreturncode: 0\nstdout:\n34 passed\nstderr:\n(empty)"
    )

    compact = main_module._compact_trace_message(message)

    assert "returncode: 0" in compact
    assert "34 passed" in compact
    assert "python -m pytest -q" in compact


def test_compact_trace_shortens_long_edit_arguments_and_diff() -> None:
    old_text = "old line\n" * 100
    message = (
        "[Step 5/20]\n\nTool: edit_file\nArguments:\n"
        + main_module.json.dumps(
            {"path": "app.py", "old_text": old_text, "new_text": "replacement"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n\nResult:\nsuccess: true\noutput:\nEdited app.py.\n"
        + ("+changed line\n" * 100)
    )

    compact = main_module._compact_trace_message(message)

    assert '"path": "app.py"' in compact
    assert "text omitted from terminal trace" in compact
    assert "Edited app.py." in compact
    assert "diff/details omitted" in compact
    assert old_text not in compact
