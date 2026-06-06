"""Data shapes internal to preprocessing.

These live here, not in :mod:`agentic_extraction.core.models`, on purpose. Core holds the
shapes that flow *between* top-level modules (a ``Chunk`` is produced here but ranked by
retrieval and cited by the agent). These shapes flow only *within* preprocessing - from
the parser to the outline and chunker - so keeping them local keeps the core vocabulary
small and stops parser-specific detail (font sizes, line positions) leaking system-wide.

They are the contract that decouples the three stages: the parser produces a
:class:`ParsedDocument`, and ``outline`` and ``chunker`` depend on that shape, never on
PyMuPDF. That is what lets those two stages be unit-tested with hand-built inputs and no
real PDF at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextLine:
    """One line of text plus the typography the outline detector reasons about.

    ``font_size`` is the largest span size on the line (headings are usually a single
    large span) and ``bold`` flags emphasised text. ``page`` is zero-based.
    """

    text: str
    page: int
    font_size: float
    bold: bool


@dataclass(frozen=True)
class TocEntry:
    """An entry from the PDF's embedded outline/bookmarks, when the author provided one.

    Preferred over typography when present, because it is the author's own structure.
    ``page`` is normalised to zero-based here.
    """

    title: str
    level: int
    page: int


@dataclass(frozen=True)
class ParsedDocument:
    """Everything the parser extracts, and the only thing the later stages consume.

    ``page_texts`` is the plain text per page (it backs ``Document.page_texts`` and the
    ``read_page`` tool). ``lines`` is the flat, reading-order line stream that the outline
    and chunker walk. ``toc`` is the embedded outline, or empty when absent.
    """

    source_path: str
    n_pages: int
    page_texts: tuple[str, ...]
    lines: tuple[TextLine, ...]
    toc: tuple[TocEntry, ...] = ()


@dataclass(frozen=True)
class Heading:
    """A detected heading, located precisely enough for the chunker to segment on it.

    ``line_index`` is the position of this heading in ``ParsedDocument.lines``, which is
    what lets the chunker slice body text between consecutive headings without re-running
    any detection of its own.
    """

    title: str
    level: int
    page: int
    line_index: int
