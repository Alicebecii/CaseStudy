"""The visual tool: render a PDF page to an image and ask a vision model about it.

The text tools cannot read figures, charts, or scanned tables; this one renders the page
and sends it to a vision LLM (DESIGN.md §3.3's visual path). It is only present when a
vision model is configured, so the default text-only system is unaffected.
"""

from __future__ import annotations

from ...core.interfaces import LLMProvider, Tool
from ...core.models import Document, ImageRef, Message, Role, ToolResult, ToolSpec
from ._shared import as_int, first


class ViewPageTool(Tool):
    """Render a page to an image and ask a vision model about it."""

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
        page = as_int(first(kwargs, "page", "page_number", "n"), default=-1)
        if not 0 <= page < self._document.n_pages:
            return ToolResult(
                content=f"Page {page} is out of range; valid pages are 0..{self._document.n_pages - 1}.",
                ok=False,
            )
        question = first(kwargs, "question", "q")
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
