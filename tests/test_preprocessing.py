"""Tests for Batch 3: preprocessing.

The outline and chunker tests build :class:`ParsedDocument` inputs by hand - no PDF - which
is the proof that those stages are decoupled from PyMuPDF. The parser and the end-to-end
loader are tested against small PDFs generated in-process with PyMuPDF, so the suite stays
self-contained and needs no checked-in fixture files.
"""

from __future__ import annotations

import pytest

from agentic_extraction.config import Settings
from agentic_extraction.core.errors import DocumentError
from agentic_extraction.preprocessing import (
    ParsedDocument,
    TextLine,
    TocEntry,
    build_outline,
    chunk,
    detect_headings,
    load_document,
    outline_to_dict,
    parse,
    section_paths,
)
from agentic_extraction.preprocessing.models import Heading

fitz = pytest.importorskip("fitz")  # PyMuPDF; a base dependency, present in any real install


# --------------------------------------------------------------- helpers / fixtures


def _body(text: str, page: int = 0) -> TextLine:
    return TextLine(text=text, page=page, font_size=11.0, bold=False)


def _heading_line(text: str, size: float, page: int = 0) -> TextLine:
    return TextLine(text=text, page=page, font_size=size, bold=True)


def _make_pdf(path, *, with_toc: bool = False) -> str:
    """Write a small two-page PDF with a title, two headings, and body text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "The Annual Report", fontsize=24)
    page1.insert_text((72, 110), "Revenue", fontsize=16)
    page1.insert_text((72, 140), "Total revenue in 2023 was 100 million lira.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Risks", fontsize=16)
    page2.insert_text((72, 110), "Currency volatility is the principal risk factor.", fontsize=11)
    if with_toc:
        # PyMuPDF TOC pages are 1-based.
        doc.set_toc([[1, "Revenue", 1], [1, "Risks", 2]])
    doc.save(str(path))
    doc.close()
    return str(path)


# ------------------------------------------------------------------------- parser


def test_parse_extracts_pages_text_and_fonts(tmp_path) -> None:
    parsed = parse(_make_pdf(tmp_path / "report.pdf"))
    assert parsed.n_pages == 2
    assert len(parsed.page_texts) == 2
    assert "100 million" in parsed.page_texts[0]
    # The title line should carry a clearly larger font than the body lines.
    sizes = {line.text: line.font_size for line in parsed.lines}
    assert sizes["The Annual Report"] > sizes["Total revenue in 2023 was 100 million lira."]


def test_parse_reads_embedded_toc(tmp_path) -> None:
    parsed = parse(_make_pdf(tmp_path / "toc.pdf", with_toc=True))
    titles = [entry.title for entry in parsed.toc]
    assert titles == ["Revenue", "Risks"]
    assert parsed.toc[1].page == 1  # normalised to zero-based


def test_parse_rejects_non_pdf(tmp_path) -> None:
    bad = tmp_path / "not.pdf"
    bad.write_text("this is plain text, not a pdf")
    with pytest.raises(DocumentError):
        parse(str(bad))


def test_parse_rejects_textless_pdf(tmp_path) -> None:
    doc = fitz.open()
    doc.new_page()  # a blank page: no text layer
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()
    with pytest.raises(DocumentError):
        parse(str(path))


# ------------------------------------------------------------------------ outline


def test_detect_headings_by_typography() -> None:
    parsed = ParsedDocument(
        source_path="x",
        n_pages=1,
        page_texts=("...",),
        lines=(
            _heading_line("Title", 24.0),
            _heading_line("Section A", 16.0),
            _body("body text one is here and is reasonably long"),
            _body("more body text to anchor the modal body font size"),
            _heading_line("Section B", 16.0),
            _body("further body content under section b for weighting"),
        ),
    )
    headings = detect_headings(parsed)
    titles = [h.title for h in headings]
    assert titles == ["Title", "Section A", "Section B"]
    # The 24pt title is a shallower level than the 16pt sections.
    assert headings[0].level == 1
    assert headings[1].level == 2 and headings[2].level == 2


def test_detect_headings_prefers_embedded_toc() -> None:
    parsed = ParsedDocument(
        source_path="x",
        n_pages=1,
        page_texts=("...",),
        lines=(_body("Revenue", page=0), _body("some prose", page=0)),
        toc=(TocEntry(title="Revenue", level=1, page=0),),
    )
    headings = detect_headings(parsed)
    assert len(headings) == 1
    assert headings[0].title == "Revenue"
    assert headings[0].line_index == 0  # located the matching line


def test_degenerate_toc_falls_back_to_typography() -> None:
    # A long document whose embedded TOC is a single title bookmark: trust typography,
    # which recovers the real sections, instead of collapsing into one flat section.
    parsed = ParsedDocument(
        source_path="x",
        n_pages=20,
        page_texts=("...",),
        lines=(
            _heading_line("The Paper Title", 22.0),
            _heading_line("Introduction", 15.0),
            _body("intro body text long enough to anchor the modal body font size here"),
            _heading_line("Method", 15.0),
            _body("method body text also long enough to weight the body font correctly"),
        ),
        toc=(TocEntry(title="The Paper Title", level=1, page=0),),
    )
    headings = detect_headings(parsed)
    assert [h.title for h in headings] == ["The Paper Title", "Introduction", "Method"]


def test_single_entry_toc_kept_for_short_document() -> None:
    # On a genuinely short document a one-entry outline is plausible; don't override it.
    parsed = ParsedDocument(
        source_path="x",
        n_pages=1,
        page_texts=("...",),
        lines=(_body("Overview", page=0), _body("some prose", page=0)),
        toc=(TocEntry(title="Overview", level=1, page=0),),
    )
    headings = detect_headings(parsed)
    assert [h.title for h in headings] == ["Overview"]


def test_build_outline_nests_by_level() -> None:
    headings = (
        Heading("Title", level=1, page=0, line_index=0),
        Heading("Section A", level=2, page=0, line_index=1),
        Heading("Section B", level=2, page=1, line_index=4),
    )
    outline = build_outline(headings)
    assert len(outline) == 1
    assert outline[0].id == "1" and outline[0].title == "Title"
    assert [c.id for c in outline[0].children] == ["1.1", "1.2"]
    assert outline[0].children[1].title == "Section B"


def test_outline_to_dict_is_json_ready_and_nested() -> None:
    headings = (
        Heading("Title", level=1, page=0, line_index=0),
        Heading("Section A", level=2, page=1, line_index=1),
    )
    outline = build_outline(headings)
    as_dict = outline_to_dict(outline)
    assert as_dict[0]["id"] == "1" and as_dict[0]["title"] == "Title"
    assert as_dict[0]["children"][0] == {
        "id": "1.1", "title": "Section A", "level": 2, "page_start": 1, "children": []
    }
    import json
    json.dumps(as_dict)  # must be serialisable without error


def test_section_paths_track_ancestry() -> None:
    headings = (
        Heading("Title", level=1, page=0, line_index=0),
        Heading("Section A", level=2, page=0, line_index=1),
        Heading("Subsection", level=3, page=0, line_index=2),
        Heading("Section B", level=2, page=1, line_index=4),
    )
    paths = section_paths(headings)
    assert paths[2] == ("Title", "Section A", "Subsection")
    assert paths[3] == ("Title", "Section B")  # popped back out of the subsection


# ------------------------------------------------------------------------ chunker


def test_chunk_tags_section_path_and_page() -> None:
    parsed = ParsedDocument(
        source_path="x",
        n_pages=2,
        page_texts=("p0", "p1"),
        lines=(
            _heading_line("Revenue", 16.0, page=0),
            _body("revenue body", page=0),
            _heading_line("Risks", 16.0, page=1),
            _body("risk body", page=1),
        ),
    )
    headings = detect_headings(parsed)
    chunks = chunk(parsed, headings, Settings())
    by_section = {c.section_path: c for c in chunks}
    assert ("Revenue",) in by_section
    assert ("Risks",) in by_section
    assert by_section[("Risks",)].page == 1  # provenance follows the section onto page 2


def test_chunk_windows_long_text_with_overlap() -> None:
    long_line = "word " * 400  # ~2000 chars, no headings -> one section
    parsed = ParsedDocument(
        source_path="x",
        n_pages=1,
        page_texts=(long_line,),
        lines=(_body(long_line.strip()),),
    )
    settings = Settings(chunk_size=800, chunk_overlap=150)
    chunks = chunk(parsed, [], settings)
    assert len(chunks) > 1  # it was split
    assert all(len(c.id) > 0 for c in chunks)
    # Consecutive windows overlap: the second starts before the first ends.
    assert chunks[0].text[-50:] in (chunks[0].text + chunks[1].text)
    assert max(len(c.text) for c in chunks) <= settings.chunk_size


def test_chunk_with_no_headings_makes_one_section() -> None:
    parsed = ParsedDocument(
        source_path="x",
        n_pages=1,
        page_texts=("short body",),
        lines=(_body("short body"),),
    )
    chunks = chunk(parsed, [], Settings())
    assert len(chunks) == 1
    assert chunks[0].section_path == ()


# ------------------------------------------------------------------ loader (e2e)


def test_load_document_end_to_end(tmp_path) -> None:
    document = load_document(_make_pdf(tmp_path / "e2e.pdf", with_toc=True), Settings())
    assert document.n_pages == 2
    assert len(document.page_texts) == 2
    assert document.chunks  # produced retrievable chunks
    assert document.outline  # produced a structure tree
    section_titles = {section.title for section in document.outline}
    assert {"Revenue", "Risks"} <= section_titles
    # Every chunk has a resolvable id and page within the document.
    for chunk_record in document.chunks:
        assert chunk_record.id
        assert 0 <= chunk_record.page < document.n_pages
