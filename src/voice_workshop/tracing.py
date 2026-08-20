"""Shared environment and LangSmith project configuration."""

from __future__ import annotations

import os


def load_environment() -> None:
    """Load a local .env file without overriding existing environment values."""
    from dotenv import load_dotenv

    load_dotenv(override=False)


def require_environment(*names: str) -> None:
    """Raise a value-free error naming any required variables that are absent."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variable(s): {joined}")


def configure_langsmith(backend: str) -> str:
    """Configure LangSmith and return the resolved project name."""
    require_environment("LANGSMITH_API_KEY")
    return f"interrupt-voice-workshop-{backend}"
