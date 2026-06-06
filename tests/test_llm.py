"""Tests for Batch 2: the LLM layer.

These cover the three backends behind one contract, plus the text tool protocol that
makes a tool-less model usable. The Ollama and OpenAI backends are exercised with
injected fakes (a fake HTTP session, a fake SDK client), so the whole suite runs offline,
deterministically, and without either optional dependency or an API key.
"""

from __future__ import annotations

import json

import pytest

from agentic_extraction.config import ProviderName, Settings
from agentic_extraction.core.errors import LLMError
from agentic_extraction.core.models import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolSpec,
)
from agentic_extraction.llm import (
    MockLLMProvider,
    OllamaLLMProvider,
    OpenAILLMProvider,
    create_llm_provider,
)
from agentic_extraction.llm.tool_protocol import (
    parse_tool_protocol_response,
    render_tools_prompt,
)

SEARCH_TOOL = ToolSpec(
    name="search",
    description="Find relevant chunks",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


# --------------------------------------------------------------------------- mock


def test_mock_default_returns_placeholder() -> None:
    provider = MockLLMProvider()
    reply = provider.complete([Message(role=Role.USER, content="hi")])
    assert reply.content == "(mock answer)"
    assert provider.calls  # the call was recorded for inspection


def test_mock_scripts_text_toolcall_and_response() -> None:
    scripted = [
        "plain answer",
        ToolCall(name="search", arguments={"query": "x"}),
        LLMResponse(content="final"),
    ]
    provider = MockLLMProvider(responses=scripted)

    first = provider.complete([])
    assert first.content == "plain answer" and not first.tool_calls

    second = provider.complete([])
    assert second.content == "" and second.tool_calls[0].name == "search"

    third = provider.complete([])
    assert third.content == "final"


def test_mock_raises_when_script_exhausted() -> None:
    provider = MockLLMProvider(responses=["only one"])
    provider.complete([])
    with pytest.raises(LLMError):
        provider.complete([])


def test_mock_handler_sees_messages_and_tools() -> None:
    def handler(messages, tools):
        return f"saw {len(messages)} msgs and {len(tools)} tools"

    provider = MockLLMProvider(handler=handler)
    reply = provider.complete([Message(role=Role.USER, content="q")], [SEARCH_TOOL])
    assert reply.content == "saw 1 msgs and 1 tools"


# ------------------------------------------------------------------- tool protocol


def test_render_tools_prompt_lists_tools_and_protocol() -> None:
    rendered = render_tools_prompt([SEARCH_TOOL])
    assert "search" in rendered
    assert '{"tool"' in rendered  # the JSON action shape is shown
    assert "query" in rendered  # parameters are described
    assert "required: query" in rendered


def test_parse_plain_text_is_final_answer() -> None:
    reply = parse_tool_protocol_response("The capital is Ankara.")
    assert reply.content == "The capital is Ankara."
    assert not reply.tool_calls


def test_parse_bare_json_is_tool_call() -> None:
    reply = parse_tool_protocol_response('{"tool": "search", "arguments": {"query": "tax"}}')
    assert not reply.content
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].arguments == {"query": "tax"}


def test_parse_fenced_json_with_prose_around_it() -> None:
    text = 'Let me look that up.\n```json\n{"tool": "search", "args": {"query": "fees"}}\n```'
    reply = parse_tool_protocol_response(text)
    assert reply.tool_calls[0].name == "search"
    # ``args`` alias is accepted, not only ``arguments``.
    assert reply.tool_calls[0].arguments == {"query": "fees"}


def test_parse_json_without_tool_key_is_treated_as_prose() -> None:
    # A JSON object that is not a tool call must not be mistaken for one.
    reply = parse_tool_protocol_response('{"answer": 42}')
    assert reply.tool_calls == ()
    assert reply.content == '{"answer": 42}'


# -------------------------------------------------------------------------- ollama


class _FakeResponse:
    def __init__(self, payload: dict, status_error: Exception | None = None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Records the last request and returns a canned response (or raises)."""

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_url: str | None = None
        self.last_json: dict | None = None

    def post(self, url, json=None, timeout=None):  # noqa: A002 - mirror requests' kwarg
        self.last_url = url
        self.last_json = json
        if self._error is not None:
            raise self._error
        return self._response


def test_ollama_builds_request_and_returns_answer() -> None:
    session = _FakeSession(_FakeResponse({"message": {"content": "  Ankara.  "}}))
    provider = OllamaLLMProvider(host="http://localhost:11434/", model="qwen2.5", session=session)

    reply = provider.complete([Message(role=Role.USER, content="capital?")])

    assert reply.content == "Ankara."  # trimmed
    assert session.last_url == "http://localhost:11434/api/chat"  # trailing slash handled
    assert session.last_json["model"] == "qwen2.5"
    assert session.last_json["stream"] is False
    assert session.last_json["messages"][-1] == {"role": "user", "content": "capital?"}


def test_ollama_with_tools_parses_tool_call_and_injects_menu() -> None:
    tool_reply = '{"tool": "search", "arguments": {"query": "interest"}}'
    session = _FakeSession(_FakeResponse({"message": {"content": tool_reply}}))
    provider = OllamaLLMProvider(session=session)

    reply = provider.complete(
        [Message(role=Role.SYSTEM, content="You are helpful."),
         Message(role=Role.USER, content="find interest")],
        tools=[SEARCH_TOOL],
    )

    assert reply.tool_calls[0].name == "search"
    # The tool menu was folded into the existing system message.
    system_sent = session.last_json["messages"][0]
    assert system_sent["role"] == "system"
    assert "You are helpful." in system_sent["content"]
    assert "Available tools" in system_sent["content"]


def test_ollama_encodes_assistant_toolcalls_and_tool_results() -> None:
    session = _FakeSession(_FakeResponse({"message": {"content": "done"}}))
    provider = OllamaLLMProvider(session=session)

    history = [
        Message(role=Role.USER, content="q"),
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(name="search", arguments={"query": "x"}),)),
        Message(role=Role.TOOL, name="search", content="chunk-1: ..."),
    ]
    provider.complete(history)

    sent = session.last_json["messages"]
    assert json.loads(sent[1]["content"]) == {"tool": "search", "arguments": {"query": "x"}}
    assert sent[2]["role"] == "user"
    assert "TOOL RESULT [search]" in sent[2]["content"]


def test_ollama_wraps_transport_errors_in_llmerror() -> None:
    session = _FakeSession(error=RuntimeError("connection refused"))
    provider = OllamaLLMProvider(session=session)
    with pytest.raises(LLMError):
        provider.complete([Message(role=Role.USER, content="q")])


def test_ollama_wraps_malformed_body_in_llmerror() -> None:
    session = _FakeSession(_FakeResponse({"unexpected": "shape"}))
    provider = OllamaLLMProvider(session=session)
    with pytest.raises(LLMError):
        provider.complete([Message(role=Role.USER, content="q")])


# -------------------------------------------------------------------------- openai


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:  # noqa: A002
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message) -> None:
        self.message = message


class _FakeCompletion:
    def __init__(self, message) -> None:
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, message, sink: dict) -> None:
        self._message = message
        self._sink = sink

    def create(self, **kwargs):
        self._sink.update(kwargs)
        return _FakeCompletion(self._message)


class _FakeClient:
    """Mimics the shape ``client.chat.completions.create(...)`` the provider uses."""

    def __init__(self, message) -> None:
        self.received: dict = {}
        completions = _FakeCompletions(message, self.received)
        self.chat = type("Chat", (), {"completions": completions})()


def test_openai_returns_plain_content() -> None:
    client = _FakeClient(_FakeMessage(content="Ankara."))
    provider = OpenAILLMProvider(model="gpt-4o-mini", client=client)
    reply = provider.complete([Message(role=Role.USER, content="capital?")])
    assert reply.content == "Ankara."
    assert reply.tool_calls == ()
    assert client.received["model"] == "gpt-4o-mini"


def test_openai_decodes_native_tool_calls() -> None:
    message = _FakeMessage(
        content=None,
        tool_calls=[_FakeToolCall("call_1", "search", '{"query": "fees"}')],
    )
    client = _FakeClient(message)
    provider = OpenAILLMProvider(client=client)

    reply = provider.complete([Message(role=Role.USER, content="find fees")], tools=[SEARCH_TOOL])

    assert reply.content == ""
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].arguments == {"query": "fees"}
    assert reply.tool_calls[0].id == "call_1"
    # Tools were sent in OpenAI's native function-tool shape.
    assert client.received["tools"][0]["function"]["name"] == "search"


def test_openai_encodes_tool_result_messages() -> None:
    client = _FakeClient(_FakeMessage(content="ok"))
    provider = OpenAILLMProvider(client=client)
    provider.complete([
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(name="search", arguments={"query": "x"}, id="call_9"),)),
        Message(role=Role.TOOL, tool_call_id="call_9", content="result text"),
    ])
    sent = client.received["messages"]
    assert sent[0]["tool_calls"][0]["id"] == "call_9"
    assert sent[1] == {"role": "tool", "tool_call_id": "call_9", "content": "result text"}


def test_openai_bad_tool_arguments_raise_llmerror() -> None:
    message = _FakeMessage(tool_calls=[_FakeToolCall("c", "search", "{not json")])
    provider = OpenAILLMProvider(client=_FakeClient(message))
    with pytest.raises(LLMError):
        provider.complete([Message(role=Role.USER, content="q")])


def test_openai_without_package_or_key_raises_llmerror() -> None:
    # No client injected and no api key -> a clear, typed failure (not an import crash).
    with pytest.raises(LLMError):
        OpenAILLMProvider(model="gpt-4o-mini", api_key="")


# ------------------------------------------------------------------------- factory


def test_factory_builds_mock_by_default() -> None:
    provider = create_llm_provider(Settings())
    assert isinstance(provider, MockLLMProvider)


def test_factory_builds_ollama_from_settings() -> None:
    settings = Settings(llm_provider=ProviderName.OLLAMA, ollama_model="qwen2.5")
    provider = create_llm_provider(settings)
    assert isinstance(provider, OllamaLLMProvider)
