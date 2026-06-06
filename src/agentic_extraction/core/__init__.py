"""Core contracts: the data shapes, interfaces, and errors everything shares.

This package depends on nothing else in the codebase. Every other module depends
on it. Importing from here gives the stable, public vocabulary of the system.
"""

from .errors import (
    AgentError,
    AgenticExtractionError,
    ConfigError,
    DocumentError,
    LLMError,
    RetrievalError,
)
from .interfaces import Embedder, LLMProvider, Retriever, Tool, Validator
from .models import (
    AgentAnswer,
    Chunk,
    Document,
    LLMResponse,
    Message,
    Role,
    ScoredChunk,
    Section,
    ToolCall,
    ToolResult,
    ToolSpec,
    Validation,
    Verdict,
)

__all__ = [
    # errors
    "AgenticExtractionError",
    "ConfigError",
    "DocumentError",
    "RetrievalError",
    "LLMError",
    "AgentError",
    # interfaces
    "LLMProvider",
    "Embedder",
    "Retriever",
    "Tool",
    "Validator",
    # models
    "Section",
    "Chunk",
    "Document",
    "ScoredChunk",
    "Role",
    "Message",
    "ToolCall",
    "LLMResponse",
    "ToolSpec",
    "ToolResult",
    "AgentAnswer",
    "Verdict",
    "Validation",
]
