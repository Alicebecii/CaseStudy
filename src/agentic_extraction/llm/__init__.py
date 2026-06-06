"""Pluggable LLM backends behind the :class:`LLMProvider` contract.

Three backends, one interface (DESIGN.md §3.4):

- :class:`MockLLMProvider`   - deterministic, offline; powers the tests and CI.
- :class:`OllamaLLMProvider` - a local model on the GPU server; free, for development.
- :class:`OpenAILLMProvider` - the OpenAI API; the paid final-verification path.

Callers should not construct these directly. Use :func:`create_llm_provider`, which reads
``Settings`` and returns the configured backend, so the choice stays in one place.
"""

from __future__ import annotations

from .factory import create_llm_provider
from .mock_provider import MockLLMProvider
from .ollama_provider import OllamaLLMProvider
from .openai_provider import OpenAILLMProvider
from .tool_protocol import (
    extract_json_object,
    parse_tool_protocol_response,
    render_tools_prompt,
)

__all__ = [
    "create_llm_provider",
    "MockLLMProvider",
    "OllamaLLMProvider",
    "OpenAILLMProvider",
    "render_tools_prompt",
    "parse_tool_protocol_response",
    "extract_json_object",
]
