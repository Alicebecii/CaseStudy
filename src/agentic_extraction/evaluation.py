"""Scoring answers against a gold question/answer set.

This is the evaluation bonus: a small, pure scorer that turns a :class:`ValidatedAnswer`
into measurable signals against a known-good expectation. It is deliberately decoupled from
how answers are produced - it reads only the frozen result shapes - so it has no LLM, no
PDF, and no I/O, and is fully unit-testable.

Gold expectations are keyed on **content, not chunk ids**: the expected facts (keywords) and
optionally the section a correct answer should cite. Chunk ids depend on chunking parameters
and would make a gold set brittle; keywords and section titles are stable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core.models import Chunk, ValidatedAnswer, Verdict

# An answer passes only if at least this fraction of the expected facts appear in it.
_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class GoldItem:
    """One gold question with the facts a correct answer must contain.

    ``expected_section`` (a section-title substring) is optional; when given, a passing
    answer must cite a chunk from that section.
    """

    question: str
    expected_keywords: tuple[str, ...]
    expected_section: str | None = None


def score(
    result: ValidatedAnswer,
    gold: GoldItem,
    chunks_by_id: dict[str, Chunk],
) -> dict[str, object]:
    """Score one answer against its gold expectation. Returns a flat dict of signals."""
    answer_text = result.answer.answer.casefold()

    matched = tuple(kw for kw in gold.expected_keywords if kw.casefold() in answer_text)
    missing = tuple(kw for kw in gold.expected_keywords if kw.casefold() not in answer_text)
    coverage = (len(matched) / len(gold.expected_keywords)) if gold.expected_keywords else 1.0

    section_match = _section_match(result.answer.cited_chunk_ids, gold.expected_section, chunks_by_id)
    grounded = result.validation.verdict is Verdict.GROUNDED

    # "passed" measures answer *accuracy* against the gold facts (expected keywords present,
    # and the right section cited when one is specified). Grounding is the validator's own
    # reliability signal and is reported separately (`grounded` / `grounded_rate`), not folded
    # into the accuracy score.
    passed = coverage >= _MIN_COVERAGE and section_match in (None, True)

    return {
        "question": gold.question,
        "grounded": grounded,
        "keyword_coverage": round(coverage, 3),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "section_match": section_match,
        "confidence": result.validation.confidence,
        "attempts": result.attempts,
        "steps": result.answer.steps,
        "passed": passed,
    }


def aggregate(scores: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarise per-question scores into aggregate metrics."""
    n = len(scores)
    if n == 0:
        return {"n": 0}

    sectioned = [s for s in scores if s["section_match"] is not None]
    return {
        "n": n,
        "pass_rate": round(_mean(bool(s["passed"]) for s in scores), 3),
        "grounded_rate": round(_mean(bool(s["grounded"]) for s in scores), 3),
        "mean_keyword_coverage": round(_mean(float(s["keyword_coverage"]) for s in scores), 3),
        "mean_confidence": round(_mean(float(s["confidence"]) for s in scores), 3),
        "mean_steps": round(_mean(float(s["steps"]) for s in scores), 3),
        "section_match_rate": (
            round(_mean(bool(s["section_match"]) for s in sectioned), 3) if sectioned else None
        ),
    }


def _section_match(
    cited_chunk_ids: Sequence[str],
    expected_section: str | None,
    chunks_by_id: dict[str, Chunk],
) -> bool | None:
    """Whether any cited chunk sits in the expected section (None if none expected)."""
    if expected_section is None:
        return None
    target = expected_section.casefold()
    for chunk_id in cited_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk and target in " ".join(chunk.section_path).casefold():
            return True
    return False


def _mean(values) -> float:
    materialised = [1.0 if value is True else 0.0 if value is False else float(value) for value in values]
    return sum(materialised) / len(materialised) if materialised else 0.0
