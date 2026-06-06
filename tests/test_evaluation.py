"""Tests for Batch A: the evaluation scorer (pure, offline)."""

from __future__ import annotations

from agentic_extraction.core.models import (
    AgentAnswer,
    Chunk,
    ValidatedAnswer,
    Validation,
    Verdict,
)
from agentic_extraction.evaluation import GoldItem, aggregate, score

CHUNKS = {"c0": Chunk(id="c0", text="...", section_path=("Results", "Framework"), page=3)}


def _result(answer: str, cited: tuple[str, ...], verdict: Verdict,
            confidence: float = 0.7, attempts: int = 1, steps: int = 2) -> ValidatedAnswer:
    return ValidatedAnswer(
        answer=AgentAnswer(question="q", answer=answer, cited_chunk_ids=cited, steps=steps),
        validation=Validation(verdict=verdict, confidence=confidence),
        attempts=attempts,
    )


def test_score_passing_answer() -> None:
    gold = GoldItem("q", ("framework", "objective"), expected_section="Framework")
    result = _result("The framework's main objective is to categorise methods.", ("c0",), Verdict.GROUNDED)
    s = score(result, gold, CHUNKS)
    assert s["grounded"] is True
    assert s["keyword_coverage"] == 1.0
    assert s["section_match"] is True
    assert s["passed"] is True


def test_score_missing_keywords_fails() -> None:
    gold = GoldItem("q", ("framework", "objective"), None)
    s = score(_result("Something unrelated entirely.", ("c0",), Verdict.GROUNDED), gold, {})
    assert s["keyword_coverage"] == 0.0
    assert s["missing_keywords"] == ("framework", "objective")
    assert s["passed"] is False


def test_grounding_reported_but_does_not_gate_accuracy() -> None:
    # An answer with the right facts passes on accuracy even if the validator's grounding
    # verdict is not GROUNDED; grounding is reported as its own separate signal.
    gold = GoldItem("q", ("framework",), None)
    s = score(_result("the framework is described here", ("c0",), Verdict.WEAK), gold, {})
    assert s["grounded"] is False     # reported
    assert s["keyword_coverage"] == 1.0
    assert s["passed"] is True        # accuracy, not grounding, gates passed


def test_score_section_mismatch_fails() -> None:
    gold = GoldItem("q", ("framework",), expected_section="Methods")
    # c0 sits in Results > Framework, not Methods.
    s = score(_result("framework", ("c0",), Verdict.GROUNDED), gold, CHUNKS)
    assert s["section_match"] is False
    assert s["passed"] is False


def test_score_no_expected_section_yields_none() -> None:
    gold = GoldItem("q", ("framework",), None)
    s = score(_result("framework", ("c0",), Verdict.GROUNDED), gold, CHUNKS)
    assert s["section_match"] is None
    assert s["passed"] is True


def test_aggregate_summarises() -> None:
    gold = GoldItem("q", ("x",), None)
    passed = score(_result("x", (), Verdict.GROUNDED), gold, {})
    failed = score(_result("y", (), Verdict.UNGROUNDED), gold, {})
    agg = aggregate([passed, failed])
    assert agg["n"] == 2
    assert agg["pass_rate"] == 0.5
    assert agg["grounded_rate"] == 0.5
    assert agg["section_match_rate"] is None  # neither item expected a section


def test_aggregate_empty_is_safe() -> None:
    assert aggregate([]) == {"n": 0}
