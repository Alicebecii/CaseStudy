"""The preprocessing facade: one PDF in, one :class:`Document` out.

This is the only function the rest of the system calls. Retrieval and the CLI depend on
``load_document`` and the ``Document`` contract, never on the parser, outline, or chunker
directly - so the three stages can be reworked freely behind this single seam.
"""

from __future__ import annotations

from ..config import Settings
from ..core.models import Document
from .chunker import chunk
from .outline import build_outline, detect_headings
from .pdf_parser import parse


def load_document(source_path: str, settings: Settings) -> Document:
    """Parse a PDF and assemble the indexed :class:`Document` the agent works over."""
    parsed = parse(source_path)
    headings = detect_headings(parsed)
    outline = build_outline(headings)
    chunks = chunk(parsed, headings, settings)
    return Document(
        source_path=parsed.source_path,
        n_pages=parsed.n_pages,
        chunks=chunks,
        outline=outline,
        page_texts=parsed.page_texts,
    )
