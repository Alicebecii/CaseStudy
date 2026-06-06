"""Deterministic grounding check - the reliability layer that needs no LLM.

An answer is only as trustworthy as the sources it cites (DESIGN.md principle #3), so this
validator asks a blunt question: does each statement in the answer actually rest on the
text of the chunks it cites? It works purely lexically - split the answer into statements,
and a statement counts as supported when enough of its content words appear in the cited
chunks' text.

Running without an LLM is the point (DESIGN.md §3.5): it is free, deterministic, works
under the mock backend, and is not the author model grading its own homework - which makes
the unit tests meaningful rather than circular. The optional LLM critic is a separate,
later layer that plugs in beside this one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.interfaces import Validator
from ..core.models import AgentAnswer, Chunk, Validation, Verdict

# Verdict thresholds on the fraction of supported statements.
_GROUNDED_MIN = 0.8
_WEAK_MIN = 0.4
# A statement is supported when at least this fraction of its content words appear in the
# cited text.
_STATEMENT_SUPPORT_MIN = 0.5

_CITATION_RE = re.compile(r"\[[A-Za-z][\w\-]*\]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")

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
    """Check that an answer's statements are lexically grounded in its cited chunks."""

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

        source_terms = set()
        for chunk in cited:
            source_terms |= _content_terms(chunk.text)

        supported: list[str] = []
        unsupported: list[str] = []
        for statement in _statements(answer.answer):
            terms = _content_terms(statement)
            if not terms:
                continue  # pure framing/punctuation, nothing to ground
            overlap = len(terms & source_terms) / len(terms)
            (supported if overlap >= _STATEMENT_SUPPORT_MIN else unsupported).append(statement)

        total = len(supported) + len(unsupported)
        fraction = (len(supported) / total) if total else 0.0
        verdict = _verdict_for(fraction)

        detail = f"{len(supported)}/{total} statements grounded in {len(cited)} cited chunk(s)."
        if missing:
            detail += f" Unresolved citations: {', '.join(missing)}."

        return Validation(
            verdict=verdict,
            confidence=round(fraction, 3),
            unsupported_claims=tuple(unsupported),
            detail=detail,
        )


def _verdict_for(fraction: float) -> Verdict:
    if fraction >= _GROUNDED_MIN:
        return Verdict.GROUNDED
    if fraction >= _WEAK_MIN:
        return Verdict.WEAK
    return Verdict.UNGROUNDED


def _statements(text: str) -> list[str]:
    """Split an answer into statements (sentences and list items), trimmed and non-empty."""
    cleaned = _CITATION_RE.sub("", text)
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [stripped for part in parts if (stripped := part.strip(" \t-•*"))]


def _content_terms(text: str) -> set[str]:
    """The set of meaningful lowercased word tokens (stop-words and single chars removed)."""
    without_citations = _CITATION_RE.sub("", text)
    terms = set()
    for token in _WORD_RE.findall(without_citations.casefold()):
        if len(token) > 1 and token not in _STOPWORDS:
            terms.add(token)
    return terms
