"""The ReAct orchestrator - the from-scratch think → act → observe loop.

We write the loop ourselves rather than use a framework (DESIGN.md §3.4) so the control
flow is fully visible and testable: the agent asks the LLM what to do, runs the tool it
asks for, feeds the result back, and repeats until the LLM answers in prose or the step
budget runs out. Because it talks to the document only through :class:`Tool`s and to the
model only through :class:`LLMProvider`, the same loop runs unchanged on the mock, Ollama,
and OpenAI backends.

Failures are observations, not crashes (DESIGN.md principle #4): an unknown tool or bad
arguments come back as a failed result the model can read and correct, and exhausting the
step budget triggers one final, tool-free answer from the evidence already gathered rather
than returning nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import Settings
from ..core.interfaces import LLMProvider, Tool
from ..core.models import (
    AgentAnswer,
    Message,
    Role,
    ToolCall,
    ToolResult,
    ToolSpec,
)

# Citations are inline [chunk_id] markers; ids look like c0, c12, etc. (leading letter).
_CITATION_RE = re.compile(r"\[([A-Za-z][\w\-]*)\]")
_ARG_PREVIEW_CHARS = 60


class ReActAgent:
    """Drives an LLM and a set of tools to a cited answer over a document."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: Sequence[Tool],
        settings: Settings,
        system_prompt: str,
    ) -> None:
        self._llm = llm
        self._tools_by_name = {tool.spec.name: tool for tool in tools}
        self._tool_specs: tuple[ToolSpec, ...] = tuple(tool.spec for tool in tools)
        self._max_steps = settings.max_steps
        self._system_prompt = system_prompt

    def answer(self, question: str) -> AgentAnswer:
        messages = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=question),
        ]
        trace: list[str] = []

        for step in range(1, self._max_steps + 1):
            response = self._llm.complete(messages, self._tool_specs)

            if not response.tool_calls:
                trace.append(f"step {step}: final answer")
                return self._finish(question, response.content, step, trace)

            messages.append(
                Message(role=Role.ASSISTANT, content=response.content, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                result = self._run_tool(call)
                status = "ok" if result.ok else "error"
                trace.append(f"step {step}: {call.name}({_preview(call.arguments)}) -> {status}")
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result.content,
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )

        # Step budget exhausted: ask once for a best-effort answer from what we have.
        trace.append(f"step {self._max_steps}: step limit reached, forcing final answer")
        messages.append(
            Message(
                role=Role.USER,
                content="You have reached the step limit. Give your best final answer now "
                "from the evidence gathered, with [chunk_id] citations, or say you could not "
                "find the answer in the document.",
            )
        )
        final = self._llm.complete(messages)
        return self._finish(question, final.content, self._max_steps, trace)

    def _finish(self, question: str, answer: str, steps: int, trace: list[str]) -> AgentAnswer:
        text = (answer or "").strip()
        return AgentAnswer(
            question=question,
            answer=text,
            cited_chunk_ids=self._extract_citations(text),
            steps=steps,
            trace=tuple(trace),
        )

    def _run_tool(self, call: ToolCall) -> ToolResult:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools_by_name)) or "(none)"
            return ToolResult(
                content=f"Unknown tool {call.name!r}. Available tools: {known}.", ok=False
            )
        try:
            return tool.run(**call.arguments)
        except Exception as exc:  # noqa: BLE001 - a tool bug becomes an observation, not a crash
            return ToolResult(content=f"Tool {call.name!r} raised: {exc}", ok=False)

    @staticmethod
    def _extract_citations(text: str) -> tuple[str, ...]:
        """Pull inline [chunk_id] citations from the answer, de-duplicated in order."""
        seen: dict[str, None] = {}
        for match in _CITATION_RE.findall(text):
            seen.setdefault(match, None)
        return tuple(seen)


def _preview(arguments: dict[str, object]) -> str:
    """A short, readable rendering of tool arguments for the trace."""
    rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return rendered[:_ARG_PREVIEW_CHARS]
