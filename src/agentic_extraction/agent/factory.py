"""The one place that wires a document, retriever, and LLM into a ready agent.

The CLI depends on this function, not on the tool classes or the orchestrator's
constructor, so how the agent is assembled stays in a single seam.
"""

from __future__ import annotations

from ..config import Settings
from ..core.interfaces import LLMProvider, Retriever
from ..core.models import Document
from .orchestrator import ReActAgent
from .prompts import build_specialist_prompt, build_system_prompt
from .tools import build_tools


def create_agent(
    document: Document,
    retriever: Retriever,
    llm: LLMProvider,
    settings: Settings,
    vision_llm: LLMProvider | None = None,
) -> ReActAgent:
    """Build a :class:`ReActAgent` with the document's tools and system prompt.

    When ``settings.enable_specialist`` is set, a focused specialist sub-agent is built
    (from the *base* tools only, so it cannot recurse) and exposed as an ``ask_specialist``
    tool. When ``vision_llm`` is given, a ``view_page`` tool is added for visual content.
    """
    valid_chunk_ids = frozenset(chunk.id for chunk in document.chunks)

    specialist: ReActAgent | None = None
    if settings.enable_specialist:
        specialist = ReActAgent(
            llm=llm,
            tools=build_tools(document, retriever, settings),  # base tools only - no recursion
            settings=settings,
            system_prompt=build_specialist_prompt(document),
            valid_chunk_ids=valid_chunk_ids,
        )

    tools = build_tools(document, retriever, settings, specialist=specialist, vision_llm=vision_llm)
    return ReActAgent(
        llm=llm,
        tools=tools,
        settings=settings,
        system_prompt=build_system_prompt(document, with_specialist=specialist is not None),
        valid_chunk_ids=valid_chunk_ids,
    )
