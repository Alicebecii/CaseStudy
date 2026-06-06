"""Deterministic grounding check - the reliability layer that needs no LLM.

An answer is only as trustworthy as the sources it cites (DESIGN.md principle #3), so this
validator asks a blunt, lexical question: how much of the answer's content vocabulary
actually appears in the chunks it cites? A grounded answer - even a paraphrased one - reuses
the source's distinctive words (names, numbers, domain terms); a fabricated one does not.

The verdict is driven by *answer-level coverage* (the fraction of the answer's content terms
found in the cited text), which credits paraphrase fairly while still catching fabrication.
Per-statement overlap is used only to *report* which specific sentences look unsupported, so
a retry has something concrete to act on.

Running without an LLM is the point (DESIGN.md §3.5): free, deterministic, works under the
mock backend, and not the author model grading its own homework. Note one honest limit: as
a lexical check it under-credits answers written in a different language than the source;
the LLM critic is the layer that bridges that gap.
"""

from __future__ import annotations

import re

from ..core.interfaces import Validator
from ..core.models import AgentAnswer, Chunk, Validation, Verdict

# Verdict thresholds on the fraction of the answer's content vocabulary found in sources.
_GROUNDED_MIN = 0.6
_WEAK_MIN = 0.3
# A statement is reported as unsupported when fewer than this fraction of its content words
# appear in the cited text.
_STATEMENT_SUPPORT_MIN = 0.3

_CITATION_RE = re.compile(r"\[[A-Za-z][\w\-]*\]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Split on sentence punctuation only when followed by whitespace (or a newline), so dotted
# tokens like "requirements.txt", "1.3.1", and "(2007)." are kept whole.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# A small stop-word set so common glue words do not inflate the overlap. Kept short on
# purpose; content words (including numbers and names) carry the grounding signal.
_STOPWORDS = frozenset(
    """
    a an the this that these those of to in on for and or but with as by at from into
    is are was were be been being it its their there here which who whom whose what
    will would can could may might must should do does did has have had not no yes
    we you they he she them his her our your my me i if then than so such also more most
    bir ve veya ile bu şu o da de için ise gibi olarak çok daha en
    """.split()
)


class GroundingValidator(Validator):
    """Check that an answer's vocabulary is lexically grounded in its cited chunks."""

    def validate(self, answer: AgentAnswer, chunks_by_id: dict[str, Chunk]) -> Validation:
        cited = [chunks_by_id[cid] for cid in answer.cited_chunk_ids if cid in chunks_by_id]
        missing = [cid for cid in answer.cited_chunk_ids if cid not in chunks_by_id]

        if not cited:
            return Validation(
                verdict=Verdict.UNGROUNDED,
                confidence=0.0,
                unsupported_claims=tuple(_statements(answer.answer)),
                detail="The answer cites no resolvable chunk; it is unsupported.",
            )

        source_terms: set[str] = set()
        for chunk in cited:
            source_terms |= _content_terms(chunk.text)

        answer_terms = _content_terms(answer.answer)
        coverage = (len(answer_terms & source_terms) / len(answer_terms)) if answer_terms else 0.0
        verdict = _verdict_for(coverage)

        unsupported = [
            statement
            for statement in _statements(answer.answer)
            if _statement_coverage(statement, source_terms) < _STATEMENT_SUPPORT_MIN
        ]

        detail = (
            f"{coverage:.0%} of the answer's content terms appear in {len(cited)} cited "
            f"chunk(s)."
        )
        if missing:
            detail += f" Unresolved citations: {', '.join(missing)}."

        return Validation(
            verdict=verdict,
            confidence=round(coverage, 3),
            unsupported_claims=tuple(unsupported),
            detail=detail,
        )


def _verdict_for(coverage: float) -> Verdict:
    if coverage >= _GROUNDED_MIN:
        return Verdict.GROUNDED
    if coverage >= _WEAK_MIN:
        return Verdict.WEAK
    return Verdict.UNGROUNDED


def _statement_coverage(statement: str, source_terms: set[str]) -> float:
    """Fraction of a statement's content terms present in the cited text (1.0 if no terms)."""
    terms = _content_terms(statement)
    if not terms:
        return 1.0  # pure framing/punctuation - nothing to ground
    return len(terms & source_terms) / len(terms)


def _statements(text: str) -> list[str]:
    """Split an answer into statements (sentences and list items), trimmed and non-empty."""
    cleaned = _CITATION_RE.sub("", text)
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [stripped for part in parts if (stripped := part.strip(" \t-•*"))]


def _content_terms(text: str) -> set[str]:
    """The set of meaningful lowercased word tokens (stop-words and single chars removed)."""
    without_citations = _CITATION_RE.sub("", text)
    return {
        token
        for token in _WORD_RE.findall(without_citations.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }
