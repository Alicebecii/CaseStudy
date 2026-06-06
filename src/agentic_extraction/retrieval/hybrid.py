"""Hybrid retrieval: combine several retrievers into one ranking via RRF.

This is the retriever the agent actually uses. It holds sub-retrievers (BM25 and dense)
and fuses their rankings with :func:`reciprocal_rank_fusion`. It *composes* - it contains
no ranking math of its own - so retrieval modes are added or removed by changing the list
passed in, never by editing this class.

Each sub-retriever is asked for a pool larger than ``top_k`` so RRF has enough overlap to
work with: a chunk that lands mid-pool in *both* lists should be able to surface above one
that tops a single list.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.interfaces import Retriever
from ..core.models import Chunk, ScoredChunk
from .fusion import reciprocal_rank_fusion

# How many candidates to pull from each sub-retriever before fusing, relative to top_k.
_MIN_POOL = 20


class HybridRetriever(Retriever):
    """Fuse the rankings of several retrievers with Reciprocal Rank Fusion."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one sub-retriever")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        pool = max(top_k, _MIN_POOL)

        ranked_id_lists: list[list[str]] = []
        chunks_by_id: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            results = retriever.search(query, pool)
            ranked_id_lists.append([scored.chunk.id for scored in results])
            for scored in results:
                chunks_by_id.setdefault(scored.chunk.id, scored.chunk)

        fused = reciprocal_rank_fusion(ranked_id_lists, self._rrf_k)
        return [
            ScoredChunk(chunk=chunks_by_id[chunk_id], score=score)
            for chunk_id, score in fused[:top_k]
        ]
