"""Document structure: find the headings, build the table-of-contents tree.

A person opens a long PDF and reads the table of contents first; we give the agent the
same affordance (DESIGN.md §3.2). Heading detection lives here, in *one* place, and its
output (a list of :class:`Heading`) is shared with the chunker, so the two never disagree
about where a section begins.

Two sources of structure, preferred in order:
1. the PDF's embedded outline, when the author provided one (it is authoritative);
2. otherwise, typography - lines whose font is larger than the body text are promoted to
   headings, and relative sizes become heading levels.

The typography path is a heuristic and can misjudge unusual layouts; that is an accepted
trade-off because retrieval is an independent path to the same content, so a missed
heading degrades navigation without ever blocking an answer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..core.models import Section
from .models import Heading, ParsedDocument, TextLine, TocEntry

# A heading line is at most this many characters; longer lines are body text even if big.
_MAX_HEADING_CHARS = 120

# A multi-page document whose embedded outline has fewer than this many entries is
# treated as having no usable outline (some PDFs ship a single bookmark - just the title).
_MIN_USABLE_TOC_ENTRIES = 2
_SHORT_DOC_PAGES = 2


def detect_headings(parsed: ParsedDocument) -> tuple[Heading, ...]:
    """Detect headings, preferring the embedded outline over typography.

    The embedded outline is authoritative *when it is usable*. Some PDFs carry a
    degenerate outline - a single bookmark holding only the title - which would collapse
    a long paper into one flat section. In that case we ignore it and fall back to
    typography, which recovers the real structure (observed on real arXiv papers).
    """
    if parsed.toc and not _toc_is_degenerate(parsed.toc, parsed):
        return _headings_from_toc(parsed)
    return _headings_from_typography(parsed)


def _toc_is_degenerate(toc: Sequence[TocEntry], parsed: ParsedDocument) -> bool:
    """True when the embedded outline is too sparse to be a real table of contents.

    A short document (a page or two) legitimately has few headings, so the sparsity test
    only applies once a document is long enough that a near-empty outline is suspect.
    """
    return len(toc) < _MIN_USABLE_TOC_ENTRIES and parsed.n_pages > _SHORT_DOC_PAGES


def outline_to_dict(sections: Sequence[Section]) -> list[dict]:
    """Serialise the section tree to a JSON-ready nested structure.

    This is the structured-outline export: ``json.dumps`` of the result is a hierarchical
    representation of the document's table of contents (id, title, level, page, children).
    """
    return [
        {
            "id": section.id,
            "title": section.title,
            "level": section.level,
            "page_start": section.page_start,
            "children": outline_to_dict(section.children),
        }
        for section in sections
    ]


def build_outline(headings: Sequence[Heading]) -> tuple[Section, ...]:
    """Build the nested :class:`Section` tree from a flat heading list.

    Section ids are hierarchical and human-readable ("1", "1.2"), which makes them stable
    references for the future ``read_section`` tool.
    """
    roots: list[_Node] = []
    stack: list[_Node] = []
    for heading in headings:
        node = _Node(title=heading.title, level=heading.level, page=heading.page)
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
    return _freeze(roots, prefix="")


def section_paths(headings: Sequence[Heading]) -> list[tuple[str, ...]]:
    """The title path (root → this heading) for each heading, in order.

    This is the chunker's view of the same structure: a chunk under a heading gets that
    heading's path as its ``section_path``, so a citation resolves to a place in the tree.
    """
    paths: list[tuple[str, ...]] = []
    stack: list[tuple[int, str]] = []
    for heading in headings:
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.title))
        paths.append(tuple(title for _, title in stack))
    return paths


# --------------------------------------------------------------------------- toc path


def _headings_from_toc(parsed: ParsedDocument) -> tuple[Heading, ...]:
    """Map embedded TOC entries to line positions so the chunker can segment on them."""
    headings: list[Heading] = []
    for entry in parsed.toc:
        line_index = _locate_line(parsed.lines, entry.title, entry.page)
        headings.append(
            Heading(title=entry.title, level=entry.level, page=entry.page, line_index=line_index)
        )
    return tuple(headings)


def _locate_line(lines: Sequence[TextLine], title: str, page: int) -> int:
    """Find the line that best matches a TOC entry; fall back to the page's first line."""
    target = title.strip().casefold()
    first_on_page = None
    for index, line in enumerate(lines):
        if line.page < page:
            continue
        if first_on_page is None:
            first_on_page = index
        if line.text.strip().casefold() == target:
            return index
        if line.page > page:
            break
    return first_on_page if first_on_page is not None else 0


# -------------------------------------------------------------------- typography path


def _headings_from_typography(parsed: ParsedDocument) -> tuple[Heading, ...]:
    """Promote lines whose font is larger than the modal body size to headings."""
    if not parsed.lines:
        return ()

    body_size = _modal_body_size(parsed.lines)
    candidates = [
        (index, line)
        for index, line in enumerate(parsed.lines)
        if line.font_size > body_size and len(line.text) <= _MAX_HEADING_CHARS
    ]
    if not candidates:
        return ()

    # Larger font -> shallower level. Distinct sizes, descending, become levels 1, 2, ...
    distinct_sizes = sorted({line.font_size for _, line in candidates}, reverse=True)
    level_of = {size: level for level, size in enumerate(distinct_sizes, start=1)}

    return tuple(
        Heading(
            title=line.text,
            level=level_of[line.font_size],
            page=line.page,
            line_index=index,
        )
        for index, line in candidates
    )


def _modal_body_size(lines: Sequence[TextLine]) -> float:
    """The most common font size, weighted by text length, taken as the body size.

    Weighting by characters (not line count) makes the estimate robust to documents with
    many short heading-like lines: the bulk of the *text* sets the body size.
    """
    weighted: Counter[float] = Counter()
    for line in lines:
        weighted[line.font_size] += max(1, len(line.text))
    return weighted.most_common(1)[0][0]


# ----------------------------------------------------------------------- tree building


class _Node:
    """A mutable tree node used only while building the immutable Section tree."""

    __slots__ = ("title", "level", "page", "children")

    def __init__(self, title: str, level: int, page: int) -> None:
        self.title = title
        self.level = level
        self.page = page
        self.children: list[_Node] = []


def _freeze(nodes: Sequence[_Node], prefix: str) -> tuple[Section, ...]:
    """Convert mutable nodes into frozen Sections, assigning hierarchical ids."""
    sections: list[Section] = []
    for position, node in enumerate(nodes, start=1):
        section_id = str(position) if not prefix else f"{prefix}.{position}"
        sections.append(
            Section(
                id=section_id,
                title=node.title,
                level=node.level,
                page_start=node.page,
                children=_freeze(node.children, prefix=section_id),
            )
        )
    return tuple(sections)
