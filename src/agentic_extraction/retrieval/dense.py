"""Dense (semantic) retrieval over embedding vectors.

Dense retrieval is the semantic half of the hybrid (DESIGN.md §3.3): it catches paraphrase
and synonymy that BM25, matching surface words, is blind to. It embeds every chunk once at
construction, then ranks by cosine similarity to the query embedding.

It depends only on the :class:`Embedder` contract and computes cosine in plain Python, so
it carries no heavy dependency itself - the model lives entirely behind the embedder. The
vector counts here (a few thousand chunks, a few hundred dims) make pure-Python cosine
perfectly fast for the MVP; swapping in numpy later is a one-file change if ever needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.interfaces import Embedder, Retriever
from ..core.models import Chunk, ScoredChunk


class DenseRetriever(Retriever):
    """Rank chunks by cosine similarity between their embeddings and the query's."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self._chunks = tuple(chunks)
        self._embedder = embedder
        vectors = embedder.embed([chunk.text for chunk in self._chunks]) if self._chunks else []
        # Pre-normalise corpus vectors so a query only needs a dot product at search time.
        self._unit_vectors = [_normalize(vector) for vector in vectors]

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_unit = _normalize(self._embedder.embed([query])[0])
        if not any(query_unit):
            return []

        scored = [
            ScoredChunk(chunk=chunk, score=_dot(query_unit, unit_vector))
            for chunk, unit_vector in zip(self._chunks, self._unit_vectors)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]


def _normalize(vector: Sequence[float]) -> list[float]:
    """Return the unit vector, or a zero vector if the input has no magnitude."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
