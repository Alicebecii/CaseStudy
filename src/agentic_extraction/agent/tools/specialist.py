"""The multi-agent tool: delegate a focused sub-question to a specialist sub-agent.

The specialist is itself a :class:`ReActAgent` (base tools + a focused prompt). This is the
multi-agent seam: the tool boundary IS the agent boundary (DESIGN.md §3.4), so a second
agent is added without touching the orchestrator loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.interfaces import Tool
from ...core.models import ToolResult, ToolSpec
from ._shared import first

if TYPE_CHECKING:
    from ..orchestrator import ReActAgent


class SpecialistAgentTool(Tool):
    """Run a sub-question through a specialist sub-agent and return its cited finding.

    The specialist is built with base tools only, so it cannot call this tool - no recursion.
    """

    def __init__(self, specialist: "ReActAgent") -> None:
        self._specialist = specialist

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ask_specialist",
            description="Delegate ONE narrow sub-question to a retrieval specialist that "
            "gathers evidence and returns a concise, cited finding. Useful for breaking a "
            "complex question into parts.",
            parameters={
                "type": "object",
                "properties": {"sub_question": {"type": "string"}},
                "required": ["sub_question"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        sub_question = first(kwargs, "sub_question", "question", "q")
        if not isinstance(sub_question, str) or not sub_question.strip():
            return ToolResult(content="ask_specialist needs a 'sub_question' string.", ok=False)
        answer = self._specialist.answer(sub_question)
        cited = ", ".join(answer.cited_chunk_ids) if answer.cited_chunk_ids else "(none)"
        return ToolResult(content=f"{answer.answer}\n\nCited: {cited}")
