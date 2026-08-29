"""Runtime configuration loaded without storing secrets in source code."""

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
COMMAND_TIMEOUT = 30.0
MAX_TOOL_OUTPUT = 12_000


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class ModelConfig:
    """Model settings read from environment variables."""

    api_key: str
    model_name: str
    base_url: str | None = None


def load_model_config(environ: Mapping[str, str] | None = None) -> ModelConfig:
    """Read and validate model configuration without exposing secret values."""
    source = os.environ if environ is None else environ

    api_key = source.get("MODEL_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("MODEL_API_KEY is required.")

    model_name = source.get("MODEL_NAME", "").strip()
    if not model_name:
        raise ConfigurationError("MODEL_NAME is required.")

    base_url = source.get("MODEL_BASE_URL", "").strip() or None
    return ModelConfig(api_key=api_key, model_name=model_name, base_url=base_url)
