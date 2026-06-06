"""Tests for Batch 6: the validator and the retry pipeline.

The grounding check is deterministic, so it is tested directly on hand-made answers. The
retry pipeline is driven by a scripted MockLLMProvider plus the real GroundingValidator, so
we assert the whole answer→validate→retry flow offline.
"""

from __future__ import annotations

from agentic_extraction.agent import create_agent
from agentic_extraction.config import Settings
from agentic_extraction.core.models import AgentAnswer, Chunk, Document, Verdict
from agentic_extraction.llm import MockLLMProvider
from agentic_extraction.retrieval import BM25Retriever
from agentic_extraction.validator import (
    GroundingValidator,
    answer_with_validation,
    create_validator,
)

CHUNKS = (
    Chunk(id="c0", text="The annual revenue grew to one hundred million lira in 2023.",
          section_path=("Revenue",), page=0),
    Chunk(id="c1", text="Currency volatility is the principal risk factor for the bank.",
          section_path=("Risks",), page=1),
)
CHUNKS_BY_ID = {c.id: c for c in CHUNKS}


def _answer(text: str, cited: tuple[str, ...]) -> AgentAnswer:
    return AgentAnswer(question="q", answer=text, cited_chunk_ids=cited, steps=1)


# ------------------------------------------------------------------ grounding check


def test_grounded_answer_is_grounded() -> None:
    answer = _answer("The annual revenue grew to one hundred million lira in 2023 [c0].", ("c0",))
    result = GroundingValidator().validate(answer, CHUNKS_BY_ID)
    assert result.verdict is Verdict.GROUNDED
    assert result.confidence >= 0.8
    assert result.unsupported_claims == ()


def test_uncited_answer_is_ungrounded() -> None:
    answer = _answer("Revenue grew to one hundred million lira.", cited=())
    result = GroundingValidator().validate(answer, CHUNKS_BY_ID)
    assert result.verdict is Verdict.UNGROUNDED
    assert result.confidence == 0.0


def test_fabricated_answer_with_real_citation_is_caught() -> None:
    # Cites a real chunk, but the statement's content is absent from that chunk's text.
    answer = _answer("The company launched a new mobile banking app in Berlin [c0].", ("c0",))
    result = GroundingValidator().validate(answer, CHUNKS_BY_ID)
    assert result.verdict is Verdict.UNGROUNDED
    assert result.unsupported_claims  # the fabricated statement is listed


def test_unresolved_citation_reported_in_detail() -> None:
    answer = _answer("Currency volatility is the principal risk [c1].", ("c1", "c99"))
    result = GroundingValidator().validate(answer, CHUNKS_BY_ID)
    assert "c99" in result.detail
    assert result.verdict is Verdict.GROUNDED  # c1 still grounds the statement


def test_create_validator_returns_grounding_validator() -> None:
    assert isinstance(create_validator(Settings()), GroundingValidator)


# ----------------------------------------------------------------------- pipeline


def _document() -> Document:
    return Document(source_path="x.pdf", n_pages=2, chunks=CHUNKS, outline=())


def _agent(llm) -> object:
    return create_agent(_document(), BM25Retriever(CHUNKS), llm, Settings())


def test_pipeline_accepts_grounded_first_answer_without_retry() -> None:
    llm = MockLLMProvider(responses=[
        "The annual revenue grew to one hundred million lira in 2023 [c0].",
    ])
    result = answer_with_validation(_agent(llm), GroundingValidator(), _document(), "revenue?")
    assert result.validation.verdict is Verdict.GROUNDED
    assert result.attempts == 1


def test_pipeline_retries_then_accepts_grounded_answer() -> None:
    # First answer cites nothing (ungrounded) -> retry -> second answer is grounded.
    llm = MockLLMProvider(responses=[
        "Revenue went up a lot.",
        "The annual revenue grew to one hundred million lira in 2023 [c0].",
    ])
    result = answer_with_validation(_agent(llm), GroundingValidator(), _document(), "revenue?")
    assert result.attempts == 2
    assert result.validation.verdict is Verdict.GROUNDED


def test_pipeline_returns_best_effort_when_retry_also_fails() -> None:
    llm = MockLLMProvider(responses=[
        "Some unsupported claim.",
        "Another unsupported claim.",
    ])
    result = answer_with_validation(_agent(llm), GroundingValidator(), _document(), "q")
    assert result.attempts == 2
    assert result.validation.verdict is not Verdict.GROUNDED  # flagged, not dressed up
