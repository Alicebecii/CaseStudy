"""The tools the agent can call to navigate and read the document.

Every way the agent touches the document is a tool (DESIGN.md principle #1), so its
behaviour is observable and testable: each step is a named call with a readable result.
Each tool is self-describing - ``spec`` tells the LLM how to call it, ``run`` executes it -
so adding a capability is one new class here plus one line in :func:`build_tools`, with no
change to the agent loop (DESIGN.md §3.4).

The four tools mirror how a person reads a long PDF: see the structure (``get_outline``),
search for evidence (``search``), read a section closely (``read_section``), or read a raw
page (``read_page``).

Argument handling is deliberately lenient - small local models name arguments slightly
differently ("id" vs "section_id"), so each tool accepts a few aliases rather than failing
a whole step on a cosmetic mismatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..config import Settings
from ..core.interfaces import LLMProvider, Retriever, Tool
from ..core.models import Chunk, Document, ImageRef, Message, Role, Section, ToolResult, ToolSpec

if TYPE_CHECKING:
    from .orchestrator import ReActAgent

# Caps so a single tool result can never blow up the model's context window.
_SECTION_CHAR_CAP = 4000
_PAGE_CHAR_CAP = 4000
_SNIPPET_CHARS = 200


def build_tools(
    document: Document,
    retriever: Retriever,
    settings: Settings,
    specialist: "ReActAgent | None" = None,
    vision_llm: LLMProvider | None = None,
) -> list[Tool]:
    """Assemble the agent's toolset for one document.

    When a ``specialist`` sub-agent is given, an ``ask_specialist`` tool is appended so the
    agent can delegate a focused sub-question to it (multi-agent, DESIGN.md §3.4). When a
    ``vision_llm`` is given, a ``view_page`` tool is appended for inspecting figures/tables.
    """
    tools: list[Tool] = [
        OutlineTool(document),
        SearchTool(retriever, settings.top_k),
        ReadSectionTool(document),
        ReadPageTool(document),
    ]
    if specialist is not None:
        tools.append(SpecialistAgentTool(specialist))
    if vision_llm is not None:
        tools.append(ViewPageTool(document, vision_llm))
    return tools


class OutlineTool(Tool):
    """Show the document's section tree - the agent's map for deciding where to look."""

    def __init__(self, document: Document) -> None:
        self._document = document

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_outline",
            description="Return the document's table of contents as an indented tree of "
            "section ids and titles. Call this first to orient yourself.",
            parameters={"type": "object", "properties": {}},
        )

    def run(self, **kwargs: object) -> ToolResult:
        if not self._document.outline:
            return ToolResult(content="(this document has no detected section structure)")
        lines: list[str] = []
        _render_outline(self._document.outline, depth=0, out=lines)
        return ToolResult(content="\n".join(lines))


class SearchTool(Tool):
    """Hybrid retrieval over the document, returning citable chunks."""

    def __init__(self, retriever: Retriever, default_top_k: int) -> None:
        self._retriever = retriever
        self._default_top_k = default_top_k

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search",
            description="Find the most relevant text chunks for a query. Each result shows "
            "its [chunk_id] (use these to cite), page, and section.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "top_k": {"type": "integer", "description": "How many results (optional)."},
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        query = _first(kwargs, "query", "q")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(content="search needs a non-empty 'query' string.", ok=False)
        top_k = _as_int(_first(kwargs, "top_k", "k"), default=self._default_top_k)

        results = self._retriever.search(query, top_k)
        if not results:
            return ToolResult(content=f"No matches for {query!r}. Try broader or different terms.")

        lines = []
        for scored in results:
            chunk = scored.chunk
            path = " > ".join(chunk.section_path) if chunk.section_path else "-"
            snippet = " ".join(chunk.text.split())[:_SNIPPET_CHARS]
            lines.append(f"[{chunk.id}] (p{chunk.page}) {path}: {snippet}")
        return ToolResult(content="\n".join(lines))


class ReadSectionTool(Tool):
    """Read the text directly under a section, and list its subsections to drill into."""

    def __init__(self, document: Document) -> None:
        self._document = document
        self._sections = _index_sections(document.outline)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_section",
            description="Read the text under a section, identified by its id from the "
            "outline (e.g. '2.1'). Also lists the section's subsections.",
            parameters={
                "type": "object",
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        section_id = _first(kwargs, "section_id", "id", "section")
        if not isinstance(section_id, str) or section_id not in self._sections:
            available = ", ".join(sorted(self._sections)) or "(none)"
            return ToolResult(
                content=f"No section with id {section_id!r}. Known ids: {available}", ok=False
            )

        section, title_path = self._sections[section_id]
        body = _section_body(self._document.chunks, title_path)
        header = f"## {section.id} {section.title} (p{section.page_start})"
        text = body if body else "(no text directly under this heading)"
        parts = [header, "", text[:_SECTION_CHAR_CAP]]
        if section.children:
            pointers = "   ".join(f"[{child.id}] {child.title}" for child in section.children)
            parts += ["", f"Subsections: {pointers}"]
        return ToolResult(content="\n".join(parts))


class ReadPageTool(Tool):
    """Read the raw text of one page - useful for tables/figures near a search hit."""

    def __init__(self, document: Document) -> None:
        self._document = document

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_page",
            description="Read the raw text of a page by its zero-based page number (as shown "
            "in search results).",
            parameters={
                "type": "object",
                "properties": {"page": {"type": "integer"}},
                "required": ["page"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        page = _as_int(_first(kwargs, "page", "page_number", "n"), default=-1)
        n_pages = self._document.n_pages
        if not 0 <= page < n_pages:
            return ToolResult(
                content=f"Page {page} is out of range; valid pages are 0..{n_pages - 1}.", ok=False
            )
        text = self._document.page_texts[page] if page < len(self._document.page_texts) else ""
        return ToolResult(content=text[:_PAGE_CHAR_CAP] if text else "(this page has no text)")


class SpecialistAgentTool(Tool):
    """Delegate a focused sub-question to a specialist sub-agent.

    The specialist is itself a :class:`ReActAgent` (base tools + a focused prompt). This is
    the multi-agent seam: the tool boundary IS the agent boundary (DESIGN.md §3.4), so a
    second agent is added without touching the orchestrator loop. The specialist is built
    with base tools only, so it cannot call this tool - no recursion.
    """

    def __init__(self, specialist: "ReActAgent") -> None:
        self._specialist = specialist

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ask_specialist",
            description="Delegate ONE narrow sub-question to a retrieval specialist that "
            "gathers evidence and returns a concise, cited finding. Useful for breaking a "
            "complex question into parts.",
            parameters={
                "type": "object",
                "properties": {"sub_question": {"type": "string"}},
                "required": ["sub_question"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        sub_question = _first(kwargs, "sub_question", "question", "q")
        if not isinstance(sub_question, str) or not sub_question.strip():
            return ToolResult(content="ask_specialist needs a 'sub_question' string.", ok=False)
        answer = self._specialist.answer(sub_question)
        cited = ", ".join(answer.cited_chunk_ids) if answer.cited_chunk_ids else "(none)"
        return ToolResult(content=f"{answer.answer}\n\nCited: {cited}")


class ViewPageTool(Tool):
    """Render a PDF page to an image and ask a vision model about it.

    The text tools cannot read figures, charts, or scanned tables; this one renders the page
    and sends it to a vision LLM (DESIGN.md §3.3's visual path). It is only present when a
    vision model is configured, so the default text-only system is unaffected.
    """

    def __init__(self, document: Document, vision_llm: LLMProvider) -> None:
        self._document = document
        self._vision_llm = vision_llm

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="view_page",
            description="Render a page to an image and ask a vision model about it - use for "
            "figures, charts, diagrams, or tables that the text tools cannot read.",
            parameters={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "Zero-based page number."},
                    "question": {"type": "string", "description": "What to look for."},
                },
                "required": ["page", "question"],
            },
        )

    def run(self, **kwargs: object) -> ToolResult:
        page = _as_int(_first(kwargs, "page", "page_number", "n"), default=-1)
        if not 0 <= page < self._document.n_pages:
            return ToolResult(
                content=f"Page {page} is out of range; valid pages are 0..{self._document.n_pages - 1}.",
                ok=False,
            )
        question = _first(kwargs, "question", "q")
        if not isinstance(question, str) or not question.strip():
            question = "Describe this page, including any figures or tables."

        try:
            image_base64 = _render_page_png(self._document.source_path, page)
        except Exception as exc:  # noqa: BLE001 - rendering failure becomes an observation
            return ToolResult(content=f"Could not render page {page}: {exc}", ok=False)

        message = Message(role=Role.USER, content=question, images=(ImageRef(data_base64=image_base64),))
        try:
            response = self._vision_llm.complete([message])
        except Exception as exc:  # noqa: BLE001 - vision backend failure becomes an observation
            return ToolResult(content=f"Vision model failed on page {page}: {exc}", ok=False)
        return ToolResult(content=response.content.strip() or "(vision model returned no text)")


def _render_page_png(source_path: str, page_number: int) -> str:
    """Render one PDF page to a base64 PNG (fitz imported lazily, as in the parser)."""
    import base64

    import fitz

    document = fitz.open(source_path)
    try:
        pixmap = document.load_page(page_number).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        png_bytes = pixmap.tobytes("png")
    finally:
        document.close()
    return base64.b64encode(png_bytes).decode()


# --------------------------------------------------------------------------- helpers


def _render_outline(sections: Sequence[Section], depth: int, out: list[str]) -> None:
    for section in sections:
        out.append("  " * depth + f"[{section.id}] {section.title} (p{section.page_start})")
        _render_outline(section.children, depth + 1, out)


def _index_sections(
    sections: Sequence[Section],
    parent_path: tuple[str, ...] = (),
) -> dict[str, tuple[Section, tuple[str, ...]]]:
    """Map each section id to (section, its full title path) for id-based lookup."""
    index: dict[str, tuple[Section, tuple[str, ...]]] = {}
    for section in sections:
        path = parent_path + (section.title,)
        index[section.id] = (section, path)
        index.update(_index_sections(section.children, path))
    return index


def _section_body(chunks: Sequence[Chunk], title_path: tuple[str, ...]) -> str:
    """Join the text of chunks that sit directly under a section (exact path match)."""
    return "\n".join(chunk.text for chunk in chunks if chunk.section_path == title_path).strip()


def _first(kwargs: dict, *names: str) -> object:
    """Return the first present keyword among ``names`` (tolerates argument aliases)."""
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


def _as_int(value: object, default: int) -> int:
    """Coerce a tool argument to int, accepting ints and numeric strings."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return default
