"""The one place that builds a retriever from a document and configuration.

The agent and CLI depend on this function and the :class:`Retriever` contract, never on
BM25 or dense directly. It also encodes the "runs out of the box" policy (DESIGN.md
principle #2): build the hybrid when dense retrieval is available, otherwise fall back to
BM25 alone - so the system works with zero setup and no model download, and gains semantic
recall automatically once the ``dense`` extra is installed.
"""

from __future__ import annotations

import importlib.util

from ..config import Settings
from ..core.interfaces import Embedder, Retriever
from ..core.models import Document
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedder import SentenceTransformerEmbedder
from .hybrid import HybridRetriever


def create_retriever(
    document: Document,
    settings: Settings,
    embedder: Embedder | None = None,
) -> Retriever:
    """Build the retriever for a document.

    Returns a :class:`HybridRetriever` (BM25 + dense) when an embedder is supplied or
    sentence-transformers is installed; otherwise a :class:`BM25Retriever`. ``embedder`` is
    injectable so tests (and callers wanting a custom model) bypass the default entirely.
    """
    chunks = document.chunks
    bm25 = BM25Retriever(chunks)

    if embedder is None and _dense_available():
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if embedder is None:
        return bm25

    dense = DenseRetriever(chunks, embedder)
    return HybridRetriever([bm25, dense], rrf_k=settings.rrf_k)


def _dense_available() -> bool:
    """True when sentence-transformers can be imported (the ``dense`` extra is installed).

    Checked with ``find_spec`` so we never trigger a model load just to decide whether
    dense retrieval is possible. Patched in tests to exercise the BM25-only fallback.
    """
    return importlib.util.find_spec("sentence_transformers") is not None
