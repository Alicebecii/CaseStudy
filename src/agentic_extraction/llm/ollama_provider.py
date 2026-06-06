"""LLM backend for a local model served by Ollama (e.g. qwen2.5 on the GPU server).

This is the free, local path: a real model with no per-token cost, which is what we use
for day-to-day development and the local end-to-end run (DESIGN.md §7). It speaks to
Ollama's ``/api/chat`` HTTP endpoint.

Tool calls go through the *text protocol* (:mod:`.tool_protocol`) rather than Ollama's
native ``tools`` field. The reason is portability and control: the text protocol works
with any model Ollama can serve, not only the few that expose native tool calling, and
the prompt/parsing is ours to tune. The cost is that we depend on the model emitting
well-formed JSON, which the lenient parser is built to tolerate.

Every failure mode here - a dead server, a timeout, a non-200, a malformed body - is
converted into one :class:`LLMError`, so callers handle one exception type instead of
the ``requests`` library's zoo (DESIGN.md principle #4: fail loudly and locally).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..core.errors import LLMError
from ..core.interfaces import LLMProvider
from ..core.models import LLMResponse, Message, Role, ToolSpec
from .tool_protocol import parse_tool_protocol_response, render_tools_prompt


class OllamaLLMProvider(LLMProvider):
    """Chat completion against a local Ollama server."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5",
        timeout: float = 120.0,
        session: object | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        # ``session`` is injectable so tests can drive the provider without a network.
        # When absent we create a real requests.Session lazily, so importing this module
        # never requires the optional ``requests`` dependency until it is actually used.
        if session is not None:
            self._session = session
        else:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - exercised via install extras
                raise LLMError(
                    "The 'requests' package is required for the Ollama backend; "
                    "install it with: pip install '.[ollama]'"
                ) from exc
            self._session = requests.Session()

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": self._encode_messages(messages, tools),
            "stream": False,
            # Greedy decoding keeps runs reproducible, which matters for a system whose
            # answers we validate and whose behaviour we want to debug deterministically.
            "options": {"temperature": 0},
        }

        try:
            response = self._session.post(
                f"{self._host}/api/chat", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - deliberately funnel every failure
            raise LLMError(f"Ollama request to {self._host} failed: {exc}") from exc

        content = _extract_content(data)
        if tools:
            return parse_tool_protocol_response(content)
        return LLMResponse(content=content.strip())

    def _encode_messages(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> list[dict[str, str]]:
        """Translate our typed messages into Ollama's wire format.

        Ollama (in this text-protocol mode) understands system/user/assistant turns of
        plain content. So we fold the tool menu into the system prompt, render the
        assistant's prior tool calls back as the JSON it "said", and feed tool results
        back as user turns the model can read.
        """
        tools_block = render_tools_prompt(tools) if tools else ""
        encoded: list[dict[str, str]] = []
        injected = False

        for message in messages:
            if message.role is Role.SYSTEM:
                content = message.content
                if tools_block and not injected:
                    content = f"{content}\n\n{tools_block}" if content else tools_block
                    injected = True
                encoded.append({"role": "system", "content": content})
            elif message.role is Role.TOOL:
                label = message.name or "tool"
                encoded.append(
                    {"role": "user", "content": f"TOOL RESULT [{label}]:\n{message.content}"}
                )
            elif message.role is Role.ASSISTANT:
                if message.tool_calls:
                    call = message.tool_calls[0]
                    content = json.dumps({"tool": call.name, "arguments": call.arguments})
                else:
                    content = message.content
                encoded.append({"role": "assistant", "content": content})
            else:  # Role.USER
                user_message: dict[str, object] = {"role": "user", "content": message.content}
                if message.images:
                    # Ollama's /api/chat takes a per-message array of raw base64 images.
                    user_message["images"] = [image.data_base64 for image in message.images]
                encoded.append(user_message)

        # If there was no system message to fold the tool menu into, add one up front.
        if tools_block and not injected:
            encoded.insert(0, {"role": "system", "content": tools_block})
        return encoded


def _extract_content(data: object) -> str:
    """Pull the assistant text out of an Ollama ``/api/chat`` response body."""
    if not isinstance(data, dict):
        raise LLMError(f"Ollama returned an unexpected body: {data!r}")
    message = data.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise LLMError(f"Ollama response missing message content: {data!r}")
    content = message["content"]
    if not isinstance(content, str):
        raise LLMError(f"Ollama message content was not text: {content!r}")
    return content
