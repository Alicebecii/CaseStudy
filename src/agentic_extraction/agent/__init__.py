"""The agent: a from-scratch ReAct orchestrator plus the tools it drives.

The orchestrator (DESIGN.md §3.4) loops think → act → observe over the document using four
tools - ``get_outline``, ``search``, ``read_section``, ``read_page`` - and produces an
:class:`AgentAnswer` with inline citations and a readable trace.

Callers should use :func:`create_agent`, which wires the tools, prompt, and orchestrator
together for one document.
"""

from __future__ import annotations

from .factory import create_agent
from .orchestrator import ReActAgent
from .tools import (
    OutlineTool,
    ReadPageTool,
    ReadSectionTool,
    SearchTool,
    SpecialistAgentTool,
    ViewPageTool,
    build_tools,
)

__all__ = [
    "create_agent",
    "ReActAgent",
    "build_tools",
    "OutlineTool",
    "SearchTool",
    "ReadSectionTool",
    "ReadPageTool",
    "SpecialistAgentTool",
    "ViewPageTool",
]
