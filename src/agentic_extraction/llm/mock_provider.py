"""A deterministic, offline LLM backend.

This is the backend that makes the rest of the system testable and runnable with no
network, no API key, and no cost (DESIGN.md principle #2). It is not a toy: it is how
the agent loop and validator are unit-tested, because a scripted sequence of replies
lets a test assert *exactly* how the agent reacts to each one.

Three ways to drive it, in order of precedence:
- ``handler``: a callable ``(messages, tools) -> reply`` for logic-driven fakes.
- ``responses``: a fixed script consumed one reply per ``complete`` call.
- neither: it returns a constant placeholder answer.

A "reply" may be an :class:`LLMResponse`, a plain ``str`` (becomes the answer text), or
a :class:`ToolCall` / list of them (becomes a tool-calling response), so tests stay terse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..core.errors import LLMError
from ..core.interfaces import LLMProvider
from ..core.models import LLMResponse, Message, ToolCall, ToolSpec

Reply = LLMResponse | str | ToolCall | Sequence[ToolCall]
Handler = Callable[[Sequence[Message], Sequence[ToolSpec]], Reply]


class MockLLMProvider(LLMProvider):
    """An :class:`LLMProvider` that returns scripted or computed replies."""

    def __init__(
        self,
        responses: Sequence[Reply] | None = None,
        handler: Handler | None = None,
        default_content: str = "(mock answer)",
    ) -> None:
        self._responses: list[Reply] | None = list(responses) if responses is not None else None
        self._handler = handler
        self._default_content = default_content
        # Every call is recorded so tests can assert what the agent actually sent.
        self.calls: list[tuple[tuple[Message, ...], tuple[ToolSpec, ...]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        self.calls.append((tuple(messages), tuple(tools)))

        if self._handler is not None:
            return _coerce(self._handler(messages, tools))

        if self._responses is not None:
            if not self._responses:
                raise LLMError("MockLLMProvider script exhausted: no more responses queued")
            return _coerce(self._responses.pop(0))

        return LLMResponse(content=self._default_content)


def _coerce(reply: Reply) -> LLMResponse:
    """Normalise the convenience reply forms into an :class:`LLMResponse`."""
    if isinstance(reply, LLMResponse):
        return reply
    if isinstance(reply, str):
        return LLMResponse(content=reply)
    if isinstance(reply, ToolCall):
        return LLMResponse(tool_calls=(reply,))
    if isinstance(reply, Sequence) and all(isinstance(item, ToolCall) for item in reply):
        return LLMResponse(tool_calls=tuple(reply))
    raise LLMError(f"MockLLMProvider cannot interpret reply of type {type(reply).__name__}")
