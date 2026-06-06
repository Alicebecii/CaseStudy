"""LLM backend for the OpenAI API, using native tool calling.

This is the paid path, reserved for the final verification run (DESIGN.md §7): a strong
hosted model to confirm the system behaves on a frontier LLM, not only on the local one.

Unlike the Ollama backend, this uses the provider's *native* tool-calling: tool specs go
in the ``tools`` field and the model returns structured ``tool_calls`` we read directly,
no text parsing. The agent sees the same :class:`LLMResponse` either way.

The OpenAI SDK is an optional dependency. We import it lazily and also allow a ``client``
to be injected, so this module imports (and unit-tests) without the package installed and
without an API key - only a real run needs both.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..core.errors import LLMError
from ..core.interfaces import LLMProvider
from ..core.models import LLMResponse, Message, Role, ToolCall, ToolSpec


class OpenAILLMProvider(LLMProvider):
    """Chat completion against the OpenAI API with native tool calling."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        client: object | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised via install extras
            raise LLMError(
                "The 'openai' package is required for the OpenAI backend; "
                "install it with: pip install '.[openai]'"
            ) from exc
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set; required for the OpenAI backend")
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [_encode_message(m) for m in messages],
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = [_encode_tool(t) for t in tools]

        try:
            completion = self._client.chat.completions.create(**kwargs)
            choice = completion.choices[0].message
        except Exception as exc:  # noqa: BLE001 - funnel SDK/network errors to one type
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        return LLMResponse(
            content=getattr(choice, "content", None) or "",
            tool_calls=_decode_tool_calls(getattr(choice, "tool_calls", None)),
        )


def _encode_message(message: Message) -> dict[str, object]:
    """Translate one typed message into the OpenAI chat format."""
    if message.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.role is Role.ASSISTANT and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for index, call in enumerate(message.tool_calls)
            ],
        }
    if message.images:
        # OpenAI vision: content becomes an array of text + image_url (data URI) blocks.
        blocks: list[dict[str, object]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for image in message.images:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image.media_type};base64,{image.data_base64}"},
            })
        return {"role": message.role.value, "content": blocks}
    return {"role": message.role.value, "content": message.content}


def _encode_tool(spec: ToolSpec) -> dict[str, object]:
    """Translate a :class:`ToolSpec` into an OpenAI ``function`` tool definition."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters or {"type": "object", "properties": {}},
        },
    }


def _decode_tool_calls(raw: object) -> tuple[ToolCall, ...]:
    """Translate the SDK's tool-call objects back into our :class:`ToolCall`s."""
    if not raw:
        return ()
    calls: list[ToolCall] = []
    for item in raw:
        function = item.function
        try:
            arguments = json.loads(function.arguments or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(
                f"OpenAI returned tool-call arguments that are not valid JSON: "
                f"{function.arguments!r}"
            ) from exc
        calls.append(
            ToolCall(name=function.name, arguments=arguments, id=getattr(item, "id", "") or "")
        )
    return tuple(calls)
