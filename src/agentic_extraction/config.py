"""Runtime configuration, read once from the environment.

All tunable behaviour lives here in one typed ``Settings`` object. Switching the
LLM backend, pointing at the remote Ollama server, or changing retrieval depth is
a single environment variable, not an edit buried in some module. The factory
reads ``Settings`` to decide which concrete implementations to build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from .core.errors import ConfigError


class ProviderName(str, Enum):
    """Which LLM backend to use."""

    MOCK = "mock"
    OLLAMA = "ollama"
    OPENAI = "openai"


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (1/true/yes/on are true), else ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Typed view of the environment.

    Defaults are chosen so the system runs out of the box with no setup: the mock
    LLM backend and a small local embedding model, neither of which needs a
    network connection or an API key.
    """

    # LLM backend selection
    llm_provider: ProviderName = ProviderName.MOCK

    # Ollama (local model on the remote server)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"

    # OpenAI (final verification run)
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Dense retrieval
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval behaviour
    top_k: int = 5
    rrf_k: int = 60

    # Agent loop
    max_steps: int = 8
    # When true, the agent gains an "ask_specialist" tool backed by a focused sub-agent.
    enable_specialist: bool = False

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables, validating as we go."""
        provider_raw = os.environ.get("LLM_PROVIDER", ProviderName.MOCK.value).lower()
        try:
            provider = ProviderName(provider_raw)
        except ValueError as exc:
            valid = ", ".join(p.value for p in ProviderName)
            raise ConfigError(
                f"LLM_PROVIDER must be one of: {valid}; got {provider_raw!r}"
            ) from exc

        return Settings(
            llm_provider=provider,
            ollama_host=os.environ.get("OLLAMA_HOST", Settings.ollama_host),
            ollama_model=os.environ.get("OLLAMA_MODEL", Settings.ollama_model),
            openai_model=os.environ.get("OPENAI_MODEL", Settings.openai_model),
            openai_api_key=os.environ.get("OPENAI_API_KEY", Settings.openai_api_key),
            embedding_model=os.environ.get("EMBEDDING_MODEL", Settings.embedding_model),
            top_k=_get_int("TOP_K", Settings.top_k),
            rrf_k=_get_int("RRF_K", Settings.rrf_k),
            max_steps=_get_int("MAX_STEPS", Settings.max_steps),
            enable_specialist=_get_bool("ENABLE_SPECIALIST", Settings.enable_specialist),
            chunk_size=_get_int("CHUNK_SIZE", Settings.chunk_size),
            chunk_overlap=_get_int("CHUNK_OVERLAP", Settings.chunk_overlap),
        )
