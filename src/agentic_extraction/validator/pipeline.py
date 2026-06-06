"""The reliability coordinator: answer, validate, and retry once if not grounded.

This composes the agent and the validator (DESIGN.md §3.5). If the first answer is not
grounded, it feeds the validator's critique back to the agent for one bounded retry - a
chance to search again or qualify its claims - then returns the best attempt. It owns no
ranking or validation logic of its own; it only orchestrates, the way the hybrid retriever
orchestrates its sub-retrievers.

Refusing to over-claim is itself a reliability feature: if even the retry is not grounded,
we still return the best attempt, but flagged with its (low) verdict rather than dressed up
as confident fact.
"""

from __future__ import annotations

from ..core.interfaces import Validator
from ..core.models import AgentAnswer, Document, ValidatedAnswer, Validation, Verdict
from ..agent.orchestrator import ReActAgent

_VERDICT_RANK = {Verdict.GROUNDED: 2, Verdict.WEAK: 1, Verdict.UNGROUNDED: 0}


def answer_with_validation(
    agent: ReActAgent,
    validator: Validator,
    document: Document,
    question: str,
    max_retries: int = 1,
) -> ValidatedAnswer:
    """Run the agent, validate, and retry once with feedback if it is not grounded."""
    chunks_by_id = {chunk.id: chunk for chunk in document.chunks}
    attempts: list[tuple[AgentAnswer, Validation]] = []
    feedback: str | None = None

    for _ in range(max_retries + 1):
        answer = agent.answer(question, feedback=feedback)
        validation = validator.validate(answer, chunks_by_id)
        attempts.append((answer, validation))
        if validation.verdict is Verdict.GROUNDED:
            break
        feedback = _feedback_from(validation)

    best_answer, best_validation = max(
        attempts, key=lambda pair: (_VERDICT_RANK[pair[1].verdict], pair[1].confidence)
    )
    return ValidatedAnswer(
        answer=best_answer, validation=best_validation, attempts=len(attempts)
    )


def _feedback_from(validation: Validation) -> str:
    """Turn a failing verdict into a corrective instruction for the next attempt."""
    lines = [
        "Your previous answer was not fully grounded in the document's cited text.",
    ]
    if validation.unsupported_claims:
        lines.append("These statements were not supported by the sources you cited:")
        lines.extend(f"- {claim}" for claim in validation.unsupported_claims)
    lines.append(
        "Search the document again for evidence, cite the exact [chunk_id]s, and drop or "
        "qualify any claim you cannot support. If the document does not contain the answer, "
        "say so plainly."
    )
    return "\n".join(lines)
