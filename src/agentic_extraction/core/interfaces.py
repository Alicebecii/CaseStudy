"""The contracts every swappable component implements.

These abstract base classes are the seams of the system. The agent depends on
``Retriever`` and ``LLMProvider``, not on the hybrid retriever or the Ollama
client. That indirection is what lets us add, remove, or rewrite an
implementation by touching only the factory, never the callers.

Each interface is intentionally tiny: one job, one method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import (
    AgentAnswer,
    Chunk,
    LLMResponse,
    MemoryRecord,
    Message,
    ScoredChunk,
    ToolResult,
    ToolSpec,
    Validation,
)


class LLMProvider(ABC):
    """A chat LLM backend.

    Implementations live in :mod:`agentic_extraction.llm` (mock, ollama, openai).
    The mock backend makes the rest of the system testable and runnable with no
    network access or cost.
    """

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        """Return the model's reply for a conversation and the tools it may call."""
        raise NotImplementedError


class Embedder(ABC):
    """Turns text into dense vectors. Used by dense retrieval."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        raise NotImplementedError


class Retriever(ABC):
    """Finds the chunks most relevant to a query."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks ranked by relevance to ``query``."""
        raise NotImplementedError


class Tool(ABC):
    """A capability the agent can invoke.

    A tool is self-describing: ``spec`` tells the LLM how to call it, and ``run``
    executes it. Adding a new agent capability means adding one ``Tool``, with no
    change to the agent loop.
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """The tool's name, description, and parameter schema for the LLM."""
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the given arguments."""
        raise NotImplementedError


class Validator(ABC):
    """Checks whether an answer is grounded in the chunks it cites."""

    @abstractmethod
    def validate(
        self,
        answer: AgentAnswer,
        chunks_by_id: dict[str, Chunk],
    ) -> Validation:
        """Return a grounding verdict for ``answer`` given the cited chunks."""
        raise NotImplementedError


class MemoryStore(ABC):
    """Cross-run memory: persists answered questions per document (extension point).

    Records the outcome of a question so a later run can recall related prior answers.
    Kept as an explicit, inspectable store rather than an opaque vector cache (DESIGN.md
    §3.6). Implementations live in :mod:`agentic_extraction.memory`.
    """

    @abstractmethod
    def record(self, record: MemoryRecord) -> None:
        """Persist one answered-question record."""
        raise NotImplementedError

    @abstractmethod
    def recall(self, document_id: str, question: str, limit: int = 3) -> list[MemoryRecord]:
        """Return prior records for a document, most relevant to ``question`` first."""
        raise NotImplementedError
