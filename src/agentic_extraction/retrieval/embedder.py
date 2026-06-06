"""The embedding backend - the one and only file that imports sentence-transformers.

This isolates the single heavy dependency of the retrieval layer, exactly as
``pdf_parser.py`` isolates PyMuPDF. Everything else in ``retrieval/`` - BM25, the dense
cosine ranking, RRF - depends only on the :class:`Embedder` contract and plain Python, so
the whole layer is testable offline with a fake embedder and adds no heavy import.

The model is local (DESIGN.md §3.3): embeddings cost nothing, need no API key, and are
reproducible by any reviewer who installs the ``dense`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import RetrievalError
from ..core.interfaces import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embed text with a local sentence-transformers model (default all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via install extras
            raise RetrievalError(
                "sentence-transformers is required for dense retrieval; "
                "install it with: pip install '.[dense]'"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - bad model name / download failure
            raise RetrievalError(f"Could not load embedding model {model_name!r}: {exc}") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(list(texts), show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - funnel encode failures to one type
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        return [list(map(float, vector)) for vector in vectors]
