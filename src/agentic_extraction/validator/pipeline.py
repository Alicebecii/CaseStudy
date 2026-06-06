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

from ..core.interfaces import MemoryStore, Validator
from ..core.models import (
    AgentAnswer,
    Document,
    MemoryRecord,
    ValidatedAnswer,
    Validation,
    Verdict,
)
from ..agent.orchestrator import ReActAgent

_VERDICT_RANK = {Verdict.GROUNDED: 2, Verdict.WEAK: 1, Verdict.UNGROUNDED: 0}


def answer_with_validation(
    agent: ReActAgent,
    validator: Validator,
    document: Document,
    question: str,
    max_retries: int = 1,
    memory: MemoryStore | None = None,
    memory_threshold: float = 0.8,
) -> ValidatedAnswer:
    """Run the agent, validate, and retry once with feedback if it is not grounded.

    When ``memory`` is given, a near-duplicate question (similarity >= ``memory_threshold``)
    that was previously answered *and grounded* is served straight from memory without
    running the agent, and every outcome is recorded for next time (DESIGN.md §3.6). With no
    memory the behaviour is exactly the single-run pipeline.
    """
    document_id = document.source_path

    if memory is not None:
        hit = memory.best_match(document_id, question)
        if (
            hit is not None
            and hit[1] >= memory_threshold
            and hit[0].verdict == Verdict.GROUNDED.value
        ):
            return _served_from_memory(hit[0], question)

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

    if memory is not None:
        memory.record(_to_memory_record(document_id, best_answer, best_validation))

    return ValidatedAnswer(
        answer=best_answer, validation=best_validation, attempts=len(attempts)
    )


def _served_from_memory(record: MemoryRecord, question: str) -> ValidatedAnswer:
    """Reconstruct a result from a recalled record; ``attempts=0`` marks 'not re-run'."""
    answer = AgentAnswer(
        question=question,
        answer=record.answer,
        cited_chunk_ids=record.cited_chunk_ids,
        steps=0,
        trace=("served from memory",),
    )
    validation = Validation(
        verdict=Verdict(record.verdict),
        confidence=record.confidence,
        detail="recalled from memory",
    )
    return ValidatedAnswer(answer=answer, validation=validation, attempts=0)


def _to_memory_record(
    document_id: str,
    answer: AgentAnswer,
    validation: Validation,
) -> MemoryRecord:
    return MemoryRecord(
        document_id=document_id,
        question=answer.question,
        answer=answer.answer,
        verdict=validation.verdict.value,
        confidence=validation.confidence,
        cited_chunk_ids=answer.cited_chunk_ids,
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
