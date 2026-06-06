"""Shared data shapes that flow between modules.

These dataclasses are the common vocabulary of the system. Preprocessing produces
them, retrieval ranks them, the agent cites them, and the validator checks them.
Keeping every record defined once, in one place, is what lets the modules stay
decoupled: a module depends on these shapes, never on another module's internals.

All records are frozen (immutable) so that data cannot be mutated in transit, which
keeps behaviour easy to reason about as the codebase changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Document structure (produced by preprocessing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """A node in the document's heading tree.

    The tree is the document's table of contents. The agent reads it first to
    decide where to look, instead of scanning the whole document.
    """

    id: str
    title: str
    level: int
    page_start: int
    children: tuple["Section", ...] = ()


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance back to the document.

    ``section_path`` lists the heading titles from the document root down to the
    section this chunk belongs to, and ``page`` is the page it starts on. Together
    they let any citation resolve back to an exact place in the source.
    """

    id: str
    text: str
    section_path: tuple[str, ...]
    page: int


@dataclass(frozen=True)
class Document:
    """A parsed PDF: its chunks, its heading tree, and per-page raw text.

    This is the single artifact preprocessing hands to the rest of the system.
    ``page_texts`` is indexed by zero-based page number and backs the
    ``read_page`` tool.
    """

    source_path: str
    n_pages: int
    chunks: tuple[Chunk, ...]
    outline: tuple[Section, ...]
    page_texts: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with the relevance score a retriever assigned to it."""

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# LLM conversation and tool calling
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Who produced a message in an LLM conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a tool with given arguments."""

    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    id: str = ""


@dataclass(frozen=True)
class Message:
    """One message in an LLM conversation.

    For ``TOOL`` role messages, ``name`` is the tool that ran and
    ``tool_call_id`` links the result back to the originating call.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    name: str = ""
    tool_call_id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """The model's reply: free text, tool calls, or both."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """Describes a tool to the LLM: its name, purpose, and parameter schema.

    ``parameters`` is a JSON Schema object so the same description works for
    native tool-calling backends and the text-protocol fallback alike.
    """

    name: str
    description: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a tool, as text the agent can read back."""

    content: str
    ok: bool = True


# ---------------------------------------------------------------------------
# Agent output and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentAnswer:
    """The agent's final answer, with provenance and a readable trace."""

    question: str
    answer: str
    cited_chunk_ids: tuple[str, ...] = ()
    steps: int = 0
    trace: tuple[str, ...] = ()


class Verdict(str, Enum):
    """How well an answer is supported by its cited sources."""

    GROUNDED = "grounded"
    WEAK = "weak"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class Validation:
    """The result of checking an answer against its cited chunks."""

    verdict: Verdict
    confidence: float
    unsupported_claims: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class ValidatedAnswer:
    """The system's terminal output: an answer paired with its grounding verdict.

    ``attempts`` records how many times the agent was run (an ungrounded first answer
    triggers a bounded retry), so a caller can see whether the answer needed correcting.
    """

    answer: AgentAnswer
    validation: Validation
    attempts: int = 1
