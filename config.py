"""Non-secret runtime configuration for the coding agent."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
COMMAND_TIMEOUT = 30.0
MAX_TOOL_OUTPUT = 12_000
