"""PDF parsing - the one and only file that touches PyMuPDF.

Isolating the library here is the whole point (DESIGN.md §3.1). Swapping PyMuPDF for
pdfplumber, or adding an OCR fallback for scanned pages, means rewriting this file and
nothing else: the outline and chunker depend on :class:`ParsedDocument`, not on ``fitz``.

Every way a PDF can refuse to be read - not a PDF, encrypted, truncated, or a scan with
no text layer - is turned into one typed :class:`DocumentError` with a clear message,
rather than a deep library stack trace (DESIGN.md principle #4).
"""

from __future__ import annotations

from ..core.errors import DocumentError
from .models import ParsedDocument, TextLine, TocEntry

# PyMuPDF span flag bit for bold text (see PyMuPDF "Text" docs: 2**4 == 16).
_BOLD_FLAG = 1 << 4


def parse(source_path: str) -> ParsedDocument:
    """Read a PDF into a :class:`ParsedDocument`.

    Raises :class:`DocumentError` if the file cannot be opened, is password-protected,
    or contains no extractable text (which usually means it is a scan needing OCR).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - declared as a base dependency
        raise DocumentError(
            "PyMuPDF is required for PDF parsing; install it with: pip install pymupdf"
        ) from exc

    try:
        doc = fitz.open(source_path)
    except Exception as exc:  # noqa: BLE001 - funnel any open failure to one type
        raise DocumentError(f"Could not open {source_path!r} as a PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise DocumentError(f"{source_path!r} is password-protected; cannot read it.")

        page_texts: list[str] = []
        lines: list[TextLine] = []
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            page_texts.append(page.get_text("text"))
            lines.extend(_lines_on_page(page, page_number))

        toc = _read_toc(doc)
        n_pages = doc.page_count
    finally:
        doc.close()

    if not "".join(page_texts).strip():
        raise DocumentError(
            f"No extractable text in {source_path!r}; the PDF appears to be scanned "
            "(OCR required)."
        )

    return ParsedDocument(
        source_path=source_path,
        n_pages=n_pages,
        page_texts=tuple(page_texts),
        lines=tuple(lines),
        toc=toc,
    )


def _lines_on_page(page: object, page_number: int) -> list[TextLine]:
    """Extract one :class:`TextLine` per visual line of text on a page.

    Uses the structured ``dict`` extraction so we get per-span font sizes and flags,
    which the outline detector needs. Image blocks are skipped here; visual content stays
    reachable by page number through ``page_texts`` / the future ``read_page`` tool.
    """
    result: list[TextLine] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 == text block; non-zero == image
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            font_size = max((span.get("size", 0.0) for span in spans), default=0.0)
            bold = any(_is_bold(span) for span in spans)
            result.append(
                TextLine(text=text, page=page_number, font_size=round(font_size, 2), bold=bold)
            )
    return result


def _is_bold(span: dict) -> bool:
    """Heuristic bold test: the bold flag bit, or 'bold' in the font name."""
    if span.get("flags", 0) & _BOLD_FLAG:
        return True
    return "bold" in str(span.get("font", "")).lower()


def _read_toc(doc: object) -> tuple[TocEntry, ...]:
    """Read the embedded outline, normalising page numbers to zero-based."""
    entries: list[TocEntry] = []
    for level, title, page in doc.get_toc():
        entries.append(TocEntry(title=title.strip(), level=level, page=max(0, page - 1)))
    return tuple(entries)
