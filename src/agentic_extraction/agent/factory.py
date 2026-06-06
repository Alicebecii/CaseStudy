"""The one place that wires a document, retriever, and LLM into a ready agent.

The CLI depends on this function, not on the tool classes or the orchestrator's
constructor, so how the agent is assembled stays in a single seam.
"""

from __future__ import annotations

from ..config import Settings
from ..core.interfaces import LLMProvider, Retriever
from ..core.models import Document
from .orchestrator import ReActAgent
from .prompts import build_system_prompt
from .tools import build_tools


def create_agent(
    document: Document,
    retriever: Retriever,
    llm: LLMProvider,
    settings: Settings,
) -> ReActAgent:
    """Build a :class:`ReActAgent` with the document's tools and system prompt."""
    tools = build_tools(document, retriever, settings)
    system_prompt = build_system_prompt(document)
    return ReActAgent(llm=llm, tools=tools, settings=settings, system_prompt=system_prompt)
