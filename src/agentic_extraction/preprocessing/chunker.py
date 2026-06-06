"""Section-aware chunking: turn the line stream into retrievable :class:`Chunk`s.

The chunker reuses the headings the outline already found, so it never re-detects
structure. It slices the line stream between consecutive headings into sections, then
windows each section into overlapping character spans of ``settings.chunk_size``.

Two properties make chunks useful downstream:
- **Provenance.** Each chunk keeps the ``section_path`` of the heading it sits under and
  the page it starts on, so any citation resolves to an exact place in the document.
- **Overlap.** Consecutive windows share ``settings.chunk_overlap`` characters, so a fact
  that straddles a window boundary still lands whole in at least one chunk.

Chunking is by characters, not tokens, on purpose: it needs no tokenizer dependency and
is fully deterministic, which keeps the base install light and tests reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..config import Settings
from ..core.models import Chunk
from .models import Heading, ParsedDocument
from .outline import section_paths


def chunk(
    parsed: ParsedDocument,
    headings: Sequence[Heading],
    settings: Settings,
) -> tuple[Chunk, ...]:
    """Split a parsed document into section-tagged, overlapping chunks."""
    chunks: list[Chunk] = []
    counter = 0
    for path, start, end in _segments(parsed, headings):
        text, offsets = _segment_text(parsed, start, end)
        if not text.strip():
            continue
        for window_start, window_text in _windows(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(
                Chunk(
                    id=f"c{counter}",
                    text=window_text,
                    section_path=path,
                    page=_page_at(offsets, window_start),
                )
            )
            counter += 1
    return tuple(chunks)


def _segments(
    parsed: ParsedDocument,
    headings: Sequence[Heading],
) -> list[tuple[tuple[str, ...], int, int]]:
    """Yield (section_path, start_line, end_line) spans covering the whole line stream.

    With no headings the whole document is one untitled section. Any text before the first
    heading becomes an untitled preamble segment so nothing is dropped.
    """
    total = len(parsed.lines)
    if not headings:
        return [((), 0, total)]

    paths = section_paths(headings)
    segments: list[tuple[tuple[str, ...], int, int]] = []

    if headings[0].line_index > 0:
        segments.append(((), 0, headings[0].line_index))

    for i, heading in enumerate(headings):
        start = heading.line_index
        end = headings[i + 1].line_index if i + 1 < len(headings) else total
        segments.append((paths[i], start, end))
    return segments


def _segment_text(
    parsed: ParsedDocument,
    start: int,
    end: int,
) -> tuple[str, list[tuple[int, int]]]:
    """Join a slice of lines into text, tracking where each line's page starts.

    ``offsets`` is a list of (char_offset, page) breakpoints, used by :func:`_page_at` to
    attribute a window back to the page it began on even when a section spans pages.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    for index in range(start, end):
        line = parsed.lines[index]
        offsets.append((position, line.page))
        parts.append(line.text)
        position += len(line.text) + 1  # +1 for the joining newline
    return "\n".join(parts), offsets


def _windows(text: str, size: int, overlap: int) -> Iterator[tuple[int, str]]:
    """Yield (start_offset, window) sliding spans of ``size`` chars overlapping ``overlap``."""
    if len(text) <= size:
        yield 0, text
        return
    step = max(1, size - overlap)
    start = 0
    while start < len(text):
        yield start, text[start : start + size]
        if start + size >= len(text):
            break
        start += step


def _page_at(offsets: Sequence[tuple[int, int]], char_position: int) -> int:
    """Return the page of the last line that began at or before ``char_position``."""
    page = offsets[0][1] if offsets else 0
    for offset, line_page in offsets:
        if offset > char_position:
            break
        page = line_page
    return page
