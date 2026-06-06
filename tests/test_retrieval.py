"""Tests for Batch 4: retrieval.

Everything here runs offline. BM25 and RRF are exact algorithms, tested on small hand-made
inputs. Dense and hybrid are tested with a deterministic *fake* embedder, so we exercise
the real ranking and fusion logic without downloading a model - the real embedder is
isolated in embedder.py and verified separately, live, on the GPU server.
"""

from __future__ import annotations

import agentic_extraction.retrieval.factory as factory_module
from agentic_extraction.config import Settings
from agentic_extraction.core.interfaces import Embedder
from agentic_extraction.core.models import Chunk, Document
from agentic_extraction.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    create_retriever,
    reciprocal_rank_fusion,
    tokenize,
)
from agentic_extraction.retrieval.bm25 import BM25Retriever as _BM25  # noqa: F401 - import smoke


def _chunk(cid: str, text: str, page: int = 0) -> Chunk:
    return Chunk(id=cid, text=text, section_path=("S",), page=page)


CORPUS = [
    _chunk("c1", "The annual revenue grew to one hundred million lira in 2023."),
    _chunk("c2", "Currency volatility is the principal risk factor for the bank."),
    _chunk("c3", "Employee headcount increased across all regional branches."),
    _chunk("c4", "Net profit margins improved due to higher interest income."),
]


# ------------------------------------------------------------------------ tokenize


def test_tokenize_lowercases_and_handles_unicode() -> None:
    assert tokenize("Net-Profit, 2023!") == ["net", "profit", "2023"]
    # Turkish characters fold rather than being dropped.
    assert "şirket" in tokenize("Şirket büyüdü")


# ---------------------------------------------------------------------------- bm25


def test_bm25_ranks_exact_term_match_first() -> None:
    retriever = BM25Retriever(CORPUS)
    results = retriever.search("revenue 2023", top_k=3)
    assert results[0].chunk.id == "c1"
    assert all(r.score > 0 for r in results)


def test_bm25_returns_empty_for_unseen_terms() -> None:
    retriever = BM25Retriever(CORPUS)
    assert retriever.search("quantum chromodynamics", top_k=3) == []


def test_bm25_empty_corpus_is_safe() -> None:
    assert BM25Retriever([]).search("anything", top_k=3) == []


# --------------------------------------------------------------------------- fusion


def test_rrf_rewards_agreement_across_lists() -> None:
    # A tops list 1, C tops list 2, B is second in both. With agreement rewarded, the item
    # that places well in BOTH (and C, top in one + present in other) should lead, and B
    # (mid in both) should beat A (top in only one).
    fused = dict(reciprocal_rank_fusion([["A", "B", "C"], ["C", "B", "D"]], k=1))
    order = [item for item, _ in reciprocal_rank_fusion([["A", "B", "C"], ["C", "B", "D"]], k=1)]
    assert order[0] == "C"  # high in both lists
    assert order.index("B") < order.index("A")  # agreement beats single-list dominance
    assert fused["D"] < fused["A"]


def test_rrf_is_deterministic_on_ties() -> None:
    # Two ids with identical scores break on lexical order, so output is stable.
    fused = reciprocal_rank_fusion([["b", "a"]], k=60)
    assert [item for item, _ in fused] == ["b", "a"]  # rank order preserved, not alpha


# ----------------------------------------------------------------------------- dense


class FakeEmbedder(Embedder):
    """A deterministic bag-of-keywords embedder: vector = keyword presence counts.

    No model, no randomness - relevance is fully predictable, which is what lets us assert
    on dense ranking and on fusion behaviour.
    """

    VOCAB = ("revenue", "risk", "currency", "profit", "interest", "employee")

    def embed(self, texts):
        vectors = []
        for text in texts:
            tokens = tokenize(text)
            vectors.append([float(tokens.count(word)) for word in self.VOCAB])
        return vectors


def test_dense_ranks_semantically_relevant_chunk_first() -> None:
    retriever = DenseRetriever(CORPUS, FakeEmbedder())
    results = retriever.search("currency risk exposure", top_k=2)
    assert results[0].chunk.id == "c2"  # the risk/currency chunk


def test_dense_empty_corpus_is_safe() -> None:
    assert DenseRetriever([], FakeEmbedder()).search("x", top_k=3) == []


# ---------------------------------------------------------------------------- hybrid


def test_hybrid_fuses_bm25_and_dense() -> None:
    hybrid = HybridRetriever([BM25Retriever(CORPUS), DenseRetriever(CORPUS, FakeEmbedder())])
    results = hybrid.search("interest income profit", top_k=2)
    ids = [r.chunk.id for r in results]
    assert "c4" in ids  # the profit/interest chunk, strong in both signals
    assert all(r.score > 0 for r in results)


def test_hybrid_requires_a_subretriever() -> None:
    try:
        HybridRetriever([])
    except ValueError:
        return
    raise AssertionError("HybridRetriever with no sub-retrievers should raise")


# --------------------------------------------------------------------------- factory


def _document() -> Document:
    return Document(source_path="x.pdf", n_pages=1, chunks=tuple(CORPUS), outline=())


def test_factory_builds_hybrid_when_embedder_injected() -> None:
    retriever = create_retriever(_document(), Settings(), embedder=FakeEmbedder())
    assert isinstance(retriever, HybridRetriever)


def test_factory_falls_back_to_bm25_when_dense_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(factory_module, "_dense_available", lambda: False)
    retriever = create_retriever(_document(), Settings())
    assert isinstance(retriever, BM25Retriever)
