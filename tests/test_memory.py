"""Tests for the cross-run memory store (DESIGN.md §3.6 extension point)."""

from __future__ import annotations

from agentic_extraction.core.models import MemoryRecord
from agentic_extraction.memory import JsonMemoryStore


def _record(doc: str, question: str, answer: str = "a") -> MemoryRecord:
    return MemoryRecord(
        document_id=doc, question=question, answer=answer, verdict="grounded", confidence=0.9,
        cited_chunk_ids=("c0",),
    )


def test_record_then_recall_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    JsonMemoryStore(path).record(_record("doc1", "What is the revenue?"))
    # A fresh instance reads what the first one wrote (it is on disk, not in memory).
    recalled = JsonMemoryStore(path).recall("doc1", "revenue")
    assert len(recalled) == 1
    assert recalled[0].question == "What is the revenue?"
    assert recalled[0].cited_chunk_ids == ("c0",)


def test_recall_filters_by_document(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "m.jsonl")
    store.record(_record("doc1", "revenue question"))
    store.record(_record("doc2", "other document question"))
    recalled = store.recall("doc2", "anything")
    assert len(recalled) == 1
    assert recalled[0].document_id == "doc2"


def test_recall_ranks_by_question_similarity(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "m.jsonl")
    store.record(_record("doc1", "How many employees work at the bank?"))
    store.record(_record("doc1", "What was the total revenue in 2023?"))
    recalled = store.recall("doc1", "tell me the revenue total for 2023", limit=1)
    assert recalled[0].question == "What was the total revenue in 2023?"


def test_recall_on_empty_store_is_safe(tmp_path) -> None:
    assert JsonMemoryStore(tmp_path / "missing.jsonl").recall("doc1", "q") == []


def test_best_match_returns_record_and_score(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "m.jsonl")
    store.record(_record("doc1", "How many employees work at the bank?"))
    store.record(_record("doc1", "What was the total revenue in 2023?"))
    hit = store.best_match("doc1", "tell me the revenue total for 2023")
    assert hit is not None
    record, similarity = hit
    assert record.question == "What was the total revenue in 2023?"
    assert 0.0 < similarity <= 1.0


def test_best_match_empty_returns_none(tmp_path) -> None:
    assert JsonMemoryStore(tmp_path / "m.jsonl").best_match("doc1", "q") is None


def test_best_match_filters_by_document(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "m.jsonl")
    store.record(_record("doc1", "revenue question"))
    assert store.best_match("doc2", "revenue question") is None
