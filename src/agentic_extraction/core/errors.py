"""Typed errors raised across the system.

A single hierarchy means a caller can catch :class:`AgenticExtractionError` to
handle anything this package raises, or catch a specific subclass to handle one
failure mode. Every module raises these instead of leaking library-specific
exceptions, so failures stay clear and contained (see the failure-handling
principle in DESIGN.md).
"""

from __future__ import annotations


class AgenticExtractionError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(AgenticExtractionError):
    """Configuration or environment is invalid (bad provider, missing key)."""


class DocumentError(AgenticExtractionError):
    """A PDF cannot be opened or parsed, or has no extractable text."""


class RetrievalError(AgenticExtractionError):
    """A retriever cannot build its index or run a query."""


class LLMError(AgenticExtractionError):
    """An LLM backend failed (network, authentication, or malformed reply)."""


class AgentError(AgenticExtractionError):
    """The agent loop cannot proceed (for example, a tool call is malformed)."""
