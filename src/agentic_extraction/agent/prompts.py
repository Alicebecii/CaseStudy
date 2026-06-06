"""The agent's system prompt - kept apart from the loop logic on purpose.

Prompt wording is the part of an agent that gets tuned most often. Isolating it here means
that churn never touches the control flow in ``orchestrator.py``: we can reword the
strategy or the citation instruction without any risk to how the loop runs.
"""

from __future__ import annotations

import os

from ..core.models import Document

_SYSTEM_TEMPLATE = """\
You are a precise question-answering agent working over a single document:
{name} ({n_pages} pages, {n_sections} top-level sections).

Answer the user's question using ONLY information found in this document. You navigate it
with tools:
- get_outline: see the section structure (start here to orient yourself).
- search: find the most relevant chunks for a query; results show [chunk_id], page, section.
- read_section: read the text under a section by its id from the outline.
- read_page: read the raw text of a page by its zero-based page number.

Work step by step: decide what you need, call one tool, read the observation, and repeat
until you have enough evidence. Use the ids and page numbers exactly as the tools show them.

When you are ready to answer, reply with plain prose and do NOT call a tool. Cite evidence
inline using the exact chunk ids shown in the search and read_section results, in square
brackets - e.g. "Net profit rose to 100M lira [c12]." Attach a [chunk_id] to a claim ONLY
when that chunk's own text directly states it; never cite a chunk that does not support the
claim. If you cannot find supporting text, search again or say the document does not contain
the answer rather than guessing."""


_SPECIALIST_TEMPLATE = """\
You are a focused retrieval specialist working over a single document:
{name} ({n_pages} pages). You are given ONE narrow sub-question by a coordinating agent.

Use the tools (get_outline, search, read_section, read_page) to find the evidence that
answers exactly that sub-question - nothing broader. Then reply with a concise finding in
plain prose, citing each fact with the exact [chunk_id] from the tool results. Do not
editorialise or restate the larger task; report only what the sources support, or say the
document does not contain it."""


_SPECIALIST_HINT = (
    "\n\nYou also have ask_specialist: for a question with several distinct parts, delegate "
    "each part as a sub_question to gather its cited evidence, then synthesise the findings - "
    "prefer this over one broad search when a question clearly has multiple parts."
)


def build_system_prompt(document: Document, with_specialist: bool = False) -> str:
    """Build the generalist system prompt, grounded in this document's basic facts.

    ``with_specialist`` appends guidance for the ``ask_specialist`` tool; it defaults to
    ``False`` so the prompt is unchanged unless a specialist sub-agent is actually present.
    """
    prompt = _SYSTEM_TEMPLATE.format(
        name=os.path.basename(document.source_path) or "document",
        n_pages=document.n_pages,
        n_sections=len(document.outline),
    )
    return prompt + _SPECIALIST_HINT if with_specialist else prompt


def build_specialist_prompt(document: Document) -> str:
    """Build the focused prompt for the evidence-gathering specialist sub-agent."""
    return _SPECIALIST_TEMPLATE.format(
        name=os.path.basename(document.source_path) or "document",
        n_pages=document.n_pages,
    )
