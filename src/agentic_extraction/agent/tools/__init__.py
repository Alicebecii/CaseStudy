"""The tools the agent can call to navigate and read the document.

Every way the agent touches the document is a tool (DESIGN.md principle #1), so its
behaviour is observable and testable: each step is a named call with a readable result.
Tools are grouped by concern, one file each:

- ``base``        the core navigation tools (get_outline · search · read_section · read_page)
- ``specialist``  the multi-agent delegation tool (ask_specialist)
- ``vision``      the visual / figure-reading tool (view_page)

Adding a capability is one new class in the relevant module plus one line in
:func:`build_tools` - the agent loop never changes (DESIGN.md §3.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import Settings
from ...core.interfaces import LLMProvider, Retriever, Tool
from ...core.models import Document
from .base import OutlineTool, ReadPageTool, ReadSectionTool, SearchTool
from .specialist import SpecialistAgentTool
from .vision import ViewPageTool

if TYPE_CHECKING:
    from ..orchestrator import ReActAgent

__all__ = [
    "build_tools",
    "OutlineTool",
    "SearchTool",
    "ReadSectionTool",
    "ReadPageTool",
    "SpecialistAgentTool",
    "ViewPageTool",
]


def build_tools(
    document: Document,
    retriever: Retriever,
    settings: Settings,
    specialist: "ReActAgent | None" = None,
    vision_llm: LLMProvider | None = None,
) -> list[Tool]:
    """Assemble the agent's toolset for one document.

    When a ``specialist`` sub-agent is given, an ``ask_specialist`` tool is appended so the
    agent can delegate a focused sub-question to it (multi-agent, DESIGN.md §3.4). When a
    ``vision_llm`` is given, a ``view_page`` tool is appended for inspecting figures/tables.
    """
    tools: list[Tool] = [
        OutlineTool(document),
        SearchTool(retriever, settings.top_k),
        ReadSectionTool(document),
        ReadPageTool(document),
    ]
    if specialist is not None:
        tools.append(SpecialistAgentTool(specialist))
    if vision_llm is not None:
        tools.append(ViewPageTool(document, vision_llm))
    return tools
