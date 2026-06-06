"""Preprocessing: PDF → parsed text → structure → chunks → :class:`Document`.

The pipeline is three small, independently testable stages behind one facade:

    parse (pdf_parser) → detect_headings/build_outline (outline) → chunk (chunker)

Only :func:`load_document` is meant for outside use; the stage functions and internal
models are exported for testing and for assembling a custom pipeline.
"""

from __future__ import annotations

from .chunker import chunk
from .loader import load_document
from .models import Heading, ParsedDocument, TextLine, TocEntry
from .outline import build_outline, detect_headings, outline_to_dict, section_paths
from .pdf_parser import parse

__all__ = [
    "load_document",
    "parse",
    "detect_headings",
    "build_outline",
    "outline_to_dict",
    "section_paths",
    "chunk",
    "ParsedDocument",
    "TextLine",
    "TocEntry",
    "Heading",
]
