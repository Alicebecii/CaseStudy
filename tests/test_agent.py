"""Tests for Batch 5: the agent.

Tools are tested against a small hand-built Document. The loop is driven by a scripted
MockLLMProvider, so we assert exactly how the orchestrator reacts to a tool call, a final
answer, an error, and the step limit - all offline and deterministic. The real ReAct run
against qwen2.5 is verified separately, live, on the GPU server.
"""

from __future__ import annotations

from agentic_extraction.agent import build_tools, create_agent
from agentic_extraction.agent.tools import (
    OutlineTool,
    ReadPageTool,
    ReadSectionTool,
    SearchTool,
)
from agentic_extraction.config import Settings
from agentic_extraction.core.models import Chunk, Document, Section, ToolCall
from agentic_extraction.llm import MockLLMProvider
from agentic_extraction.retrieval import BM25Retriever


def _document() -> Document:
    outline = (
        Section(id="1", title="Introduction", level=1, page_start=0),
        Section(
            id="2",
            title="Methods",
            level=1,
            page_start=1,
            children=(
                Section(id="2.1", title="Data", level=2, page_start=1),
                Section(id="2.2", title="Model", level=2, page_start=2),
            ),
        ),
    )
    chunks = (
        Chunk(id="c0", text="The introduction discusses revenue growth.", section_path=("Introduction",), page=0),
        Chunk(id="c1", text="Methods overview paragraph.", section_path=("Methods",), page=1),
        Chunk(id="c2", text="The dataset has 10000 labelled samples.", section_path=("Methods", "Data"), page=1),
        Chunk(id="c3", text="The model is a transformer encoder.", section_path=("Methods", "Model"), page=2),
    )
    return Document(
        source_path="/tmp/paper.pdf",
        n_pages=3,
        chunks=chunks,
        outline=outline,
        page_texts=("Intro page about revenue.", "Methods overview. 10000 samples.", "Transformer encoder."),
    )


# ----------------------------------------------------------------------------- tools


def test_outline_tool_renders_ids_and_titles() -> None:
    content = OutlineTool(_document()).run().content
    assert "[1] Introduction (p0)" in content
    assert "[2.1] Data" in content  # nested and indented


def test_search_tool_formats_citable_hits() -> None:
    tool = SearchTool(BM25Retriever(_document().chunks), default_top_k=3)
    content = tool.run(query="dataset samples").content
    assert "[c2]" in content and "(p1)" in content
    assert "Methods > Data" in content


def test_search_tool_requires_query() -> None:
    tool = SearchTool(BM25Retriever(_document().chunks), default_top_k=3)
    assert tool.run().ok is False


def test_read_section_returns_body_and_subsections() -> None:
    result = ReadSectionTool(_document()).run(section_id="2")
    assert "Methods overview paragraph." in result.content
    assert "[2.1] Data" in result.content and "[2.2] Model" in result.content


def test_read_section_accepts_id_alias_and_rejects_unknown() -> None:
    tool = ReadSectionTool(_document())
    assert "10000 labelled" in tool.run(id="2.1").content  # 'id' alias works
    assert tool.run(section_id="99").ok is False


def test_read_page_bounds_checked() -> None:
    tool = ReadPageTool(_document())
    assert "Transformer encoder." in tool.run(page=2).content
    assert tool.run(page=9).ok is False
    assert "10000 samples" in tool.run(page="1").content  # numeric string coerced


def test_build_tools_exposes_the_four_tools() -> None:
    tools = build_tools(_document(), BM25Retriever(_document().chunks), Settings())
    assert {t.spec.name for t in tools} == {"get_outline", "search", "read_section", "read_page"}


# ---------------------------------------------------------------------- orchestrator


def _agent(llm, **overrides) -> object:
    settings = Settings(**overrides)
    return create_agent(_document(), BM25Retriever(_document().chunks), llm, settings)


def test_agent_runs_tool_then_answers_with_citations() -> None:
    llm = MockLLMProvider(responses=[
        ToolCall(name="search", arguments={"query": "dataset"}),
        "The dataset has 10000 labelled samples [c2].",
    ])
    answer = _agent(llm).answer("How big is the dataset?")
    assert answer.answer == "The dataset has 10000 labelled samples [c2]."
    assert answer.cited_chunk_ids == ("c2",)
    assert answer.steps == 2
    assert any("search" in line for line in answer.trace)


def test_agent_answers_immediately_without_tools() -> None:
    llm = MockLLMProvider(responses=["It is a transformer encoder [c3]."])
    answer = _agent(llm).answer("What model is used?")
    assert answer.cited_chunk_ids == ("c3",)
    assert answer.steps == 1


def test_agent_feeds_unknown_tool_error_back_and_recovers() -> None:
    llm = MockLLMProvider(responses=[
        ToolCall(name="nonexistent", arguments={}),
        "Sorry, here is the answer anyway [c0].",
    ])
    answer = _agent(llm).answer("anything")
    assert answer.cited_chunk_ids == ("c0",)
    assert any("error" in line for line in answer.trace)


def test_agent_forces_final_answer_when_step_budget_exhausted() -> None:
    # Every step returns a tool call, never a prose answer, so the budget runs out and the
    # orchestrator issues one final tool-free completion.
    llm = MockLLMProvider(responses=[
        ToolCall(name="search", arguments={"query": "x"}),
        "Best effort final answer [c1].",
    ])
    answer = _agent(llm, max_steps=1).answer("question")
    assert answer.steps == 1
    assert answer.answer == "Best effort final answer [c1]."
    assert any("step limit" in line for line in answer.trace)


def test_agent_dedupes_citations_in_order() -> None:
    llm = MockLLMProvider(responses=["A [c3] then B [c0] then again [c3]."])
    answer = _agent(llm).answer("q")
    assert answer.cited_chunk_ids == ("c3", "c0")


def test_agent_resolves_chunk_ids_regardless_of_formatting() -> None:
    # The model wraps the id ("[Chunk c2]") and adds a page ref ("[p9]") and a section ref
    # ("[1.3.1]"). Only the real chunk id should resolve.
    llm = MockLLMProvider(responses=["See [Chunk c2] on [p9] in section [1.3.1]."])
    answer = _agent(llm).answer("q")
    assert answer.cited_chunk_ids == ("c2",)
