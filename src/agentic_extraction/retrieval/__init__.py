"""Retrieval: find the chunks most relevant to a query.

A hybrid of two complementary signals fused by Reciprocal Rank Fusion (DESIGN.md §3.3):

- :class:`BM25Retriever`   - lexical; exact terms, names, numbers.
- :class:`DenseRetriever`  - semantic; paraphrase and synonymy (via an :class:`Embedder`).
- :class:`HybridRetriever` - fuses the two with :func:`reciprocal_rank_fusion`.

Callers should use :func:`create_retriever`, which assembles the right retriever for the
installed dependencies and returns it behind the :class:`Retriever` contract.
"""

from __future__ import annotations

from .bm25 import BM25Retriever, tokenize
from .dense import DenseRetriever
from .embedder import SentenceTransformerEmbedder
from .factory import create_retriever
from .fusion import reciprocal_rank_fusion
from .hybrid import HybridRetriever

__all__ = [
    "create_retriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "SentenceTransformerEmbedder",
    "reciprocal_rank_fusion",
    "tokenize",
]
