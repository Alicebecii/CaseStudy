"""The one place that turns configuration into a concrete LLM backend.

The agent and validator depend on the :class:`LLMProvider` contract, never on a specific
backend. This factory is the single seam where ``Settings`` is read and a concrete
provider is chosen, so switching backends is one environment variable and adding a new
backend touches only this function (DESIGN.md §3.4, principle #2).
"""

from __future__ import annotations

from ..config import ProviderName, Settings
from ..core.errors import ConfigError
from ..core.interfaces import LLMProvider
from .mock_provider import MockLLMProvider
from .ollama_provider import OllamaLLMProvider
from .openai_provider import OpenAILLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the LLM backend named by ``settings.llm_provider``."""
    provider = settings.llm_provider

    if provider is ProviderName.MOCK:
        return MockLLMProvider()
    if provider is ProviderName.OLLAMA:
        return OllamaLLMProvider(host=settings.ollama_host, model=settings.ollama_model)
    if provider is ProviderName.OPENAI:
        return OpenAILLMProvider(model=settings.openai_model, api_key=settings.openai_api_key)

    # Unreachable while ProviderName stays exhaustive; kept so a new enum member that is
    # added without a branch here fails loudly instead of silently returning nothing.
    raise ConfigError(f"No LLM backend wired for provider {provider!r}")
