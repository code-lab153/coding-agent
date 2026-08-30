"""Small command-line entry point for the coding agent."""

import argparse
from collections.abc import Sequence
import sys

from agent import Agent, AgentError
from config import ConfigurationError, WORKSPACE_ROOT
from llm import LLMClient, LLMError


def _build_parser() -> argparse.ArgumentParser:
    """Create the intentionally small command-line interface."""
    parser = argparse.ArgumentParser(description="Run the local coding agent.")
    parser.add_argument("task", nargs="?", help="Programming task for the agent.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble the application, run one task, and return a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        task = args.task if args.task is not None else input("Task: ")
        if not task.strip():
            print("Error: task must not be empty.", file=sys.stderr)
            return 2

        print("Coding Agent")
        print(f"Workspace: {WORKSPACE_ROOT.name}")
        print()

        agent = Agent(LLMClient(), trace=print)
        final_answer = agent.run(task)
        print("\n[Final]")
        print(final_answer)
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
