"""A text-based tool-call protocol for LLM backends without native tool calling.

Native tool-calling APIs (OpenAI) return tool invocations as structured data. Small
local models served over Ollama do not, so we give them a *protocol*: the tool menu
is rendered into the prompt, the model replies with a single JSON object naming the
tool and its arguments, and we parse that back into a :class:`ToolCall`.

Putting this here, behind the provider, is what lets the agent stay backend-agnostic:
:meth:`LLMProvider.complete` always returns an :class:`LLMResponse` with ``tool_calls``,
whether those came from a native API or from parsing text. The agent never knows which.

Parsing is deliberately lenient. A small model may wrap its JSON in a code fence, add a
sentence before it, or use ``args``/``parameters`` instead of ``arguments``. We accept
all of these rather than fail a whole step on a cosmetic deviation; only when no tool
object can be recovered do we treat the reply as a final, plain-text answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..core.models import LLMResponse, ToolCall, ToolSpec

# Keys a model might plausibly use for the tool name and its arguments. We read the
# first one present, so a slightly off-spec reply still parses.
_NAME_KEYS = ("tool", "name", "action")
_ARG_KEYS = ("arguments", "args", "parameters", "input")

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def render_tools_prompt(tools: Sequence[ToolSpec]) -> str:
    """Render the tool menu and the JSON-action instructions for the system prompt.

    The contract we ask the model to follow is intentionally tiny: emit *exactly one*
    JSON object to call a tool, or plain prose to give the final answer. A small model
    follows a small contract more reliably than a verbose one.
    """
    lines = [
        "You have tools you may call to gather information before answering.",
        "",
        "To CALL A TOOL, reply with ONLY a JSON object and nothing else:",
        '    {"tool": "<tool_name>", "arguments": {<arg>: <value>, ...}}',
        "",
        "To give your FINAL ANSWER, reply with plain text (no JSON object).",
        "",
        "Available tools:",
    ]
    for spec in tools:
        lines.append(f"- {spec.name}: {spec.description}")
        params = spec.parameters or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if props:
            rendered = ", ".join(
                f"{key} ({(meta or {}).get('type', 'any')})"
                for key, meta in props.items()
            )
            required = params.get("required") or []
            suffix = f" [required: {', '.join(required)}]" if required else ""
            lines.append(f"    arguments: {rendered}{suffix}")
    return "\n".join(lines)


def parse_tool_protocol_response(text: str) -> LLMResponse:
    """Turn a model's raw text into an :class:`LLMResponse`.

    If a tool-call object can be recovered, return it as a ``tool_calls`` entry with no
    content. Otherwise the whole text is the final answer.
    """
    obj = extract_json_object(text)
    if obj is not None:
        name = _first_present(obj, _NAME_KEYS)
        if isinstance(name, str) and name.strip():
            args = _first_present(obj, _ARG_KEYS)
            if not isinstance(args, dict):
                args = {}
            call = ToolCall(name=name.strip(), arguments=args)
            return LLMResponse(content="", tool_calls=(call,))
    return LLMResponse(content=text.strip())


def _first_present(obj: dict, keys: Sequence[str]) -> object:
    """Return the value of the first key in ``keys`` that ``obj`` actually has."""
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from free-form model text.

    Tries a fenced ```json block first, then the first brace-balanced ``{...}`` span.
    Returns ``None`` if nothing parses to a dict. Shared by the text tool protocol (to
    spot a tool call) and the LLM critic (to read its JSON verdict), so the lenient
    parsing lives in one place.
    """
    candidates: list[str] = []

    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))

    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` substring, or ``None``.

    A simple depth counter is enough here and avoids pulling in a JSON streaming
    parser. Braces inside strings are rare in tool calls and, if they trip the
    counter, the candidate simply fails ``json.loads`` and we fall back to prose.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
