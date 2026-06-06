"""Cross-run memory store - the extension point from DESIGN.md §3.6.

A deliberately simple, inspectable store: each answered question is appended as one JSON
line keyed by document, and ``recall`` returns prior records for that document ranked by
lexical overlap with the new question. It is intentionally *not* an opaque vector cache, so
its effect on answers stays auditable (DESIGN.md principle #4).

This realises the documented seam - it is functional and tested on its own. Wiring it into
the agent loop (short-cutting near-duplicate questions, seeding retrieval with chunks that
previously answered related questions) is the cross-task-learning bonus; the store is the
foundation that makes it a small addition rather than a redesign.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.interfaces import MemoryStore
from ..core.models import MemoryRecord

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class JsonMemoryStore(MemoryStore):
    """An append-only, line-delimited JSON memory store on disk."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(self, record: MemoryRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")

    def recall(self, document_id: str, question: str, limit: int = 3) -> list[MemoryRecord]:
        for_document = [r for r in self._read() if r.document_id == document_id]
        question_terms = _terms(question)
        for_document.sort(
            key=lambda record: _overlap(question_terms, _terms(record.question)),
            reverse=True,
        )
        return for_document[:limit]

    def best_match(self, document_id: str, question: str) -> tuple[MemoryRecord, float] | None:
        question_terms = _terms(question)
        best: tuple[MemoryRecord, float] | None = None
        for record in self._read():
            if record.document_id != document_id:
                continue
            similarity = _overlap(question_terms, _terms(record.question))
            if best is None or similarity > best[1]:
                best = (record, similarity)
        return best

    def _read(self) -> list[MemoryRecord]:
        if not self._path.exists():
            return []
        records: list[MemoryRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # skip a corrupt line rather than fail the whole recall
        return records


def _to_dict(record: MemoryRecord) -> dict:
    return {
        "document_id": record.document_id,
        "question": record.question,
        "answer": record.answer,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "cited_chunk_ids": list(record.cited_chunk_ids),
    }


def _from_dict(data: dict) -> MemoryRecord:
    return MemoryRecord(
        document_id=data["document_id"],
        question=data["question"],
        answer=data.get("answer", ""),
        verdict=data.get("verdict", ""),
        confidence=float(data.get("confidence", 0.0)),
        cited_chunk_ids=tuple(data.get("cited_chunk_ids", ())),
    )


def _terms(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.casefold()) if len(token) > 1}


def _overlap(a: set[str], b: set[str]) -> float:
    """Jaccard overlap between two term sets (0 when either is empty)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
