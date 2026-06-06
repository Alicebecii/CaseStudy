"""LLM critic - an independent second opinion on whether an answer is grounded.

This is the second validation layer (DESIGN.md §3.5). Where the deterministic check is
lexical, the critic reads the cited sources and the answer and judges *entailment*: is each
claim actually supported by these passages? It is a genuine second opinion because it sees
only the question, the answer, and the cited text - never the agent's own reasoning - and
runs from a different, adversarial prompt.

It is advisory by design: the deterministic check is the floor (it always runs), so the
critic only needs to *add* catches. Accordingly, if the model errors or returns something
unparseable, the critic raises no objection rather than vetoing a possibly-fine answer -
the deterministic verdict still stands underneath.
"""

from __future__ import annotations

from ..core.errors import LLMError
from ..core.interfaces import LLMProvider, Validator
from ..core.models import AgentAnswer, Chunk, Message, Role, Validation, Verdict
from ..llm.tool_protocol import extract_json_object

# How a verdict word maps to a verdict, and a verdict to a representative confidence.
_VERDICT_WORDS = {
    "grounded": Verdict.GROUNDED, "supported": Verdict.GROUNDED, "entailed": Verdict.GROUNDED,
    "yes": Verdict.GROUNDED, "weak": Verdict.WEAK, "partial": Verdict.WEAK,
    "partially": Verdict.WEAK, "ungrounded": Verdict.UNGROUNDED, "unsupported": Verdict.UNGROUNDED,
    "no": Verdict.UNGROUNDED,
}
_CONFIDENCE = {Verdict.GROUNDED: 0.9, Verdict.WEAK: 0.5, Verdict.UNGROUNDED: 0.15}
_MAX_CLAIMS = 10

_SYSTEM_PROMPT = """\
You are a strict fact-checker. You are given a QUESTION, the SOURCE passages an answer
cited, and the ANSWER itself. Decide whether every factual claim in the ANSWER is supported
by the SOURCES alone - do not use any outside knowledge.

Reply with ONLY a JSON object, no other text:
{"verdict": "grounded" | "weak" | "ungrounded", "unsupported_claims": ["..."]}
where "grounded" = every claim is supported, "weak" = some claims are unsupported, and
"ungrounded" = most or all claims are unsupported. List each unsupported statement."""


class LLMCriticValidator(Validator):
    """Judge an answer's grounding with an LLM second opinion."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def validate(self, answer: AgentAnswer, chunks_by_id: dict[str, Chunk]) -> Validation:
        cited = [chunks_by_id[cid] for cid in answer.cited_chunk_ids if cid in chunks_by_id]
        if not cited:
            return Validation(
                verdict=Verdict.UNGROUNDED,
                confidence=0.0,
                unsupported_claims=(answer.answer,) if answer.answer else (),
                detail="critic: the answer cites no resolvable source.",
            )

        try:
            response = self._llm.complete(
                [
                    Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
                    Message(role=Role.USER, content=_build_user_prompt(answer, cited)),
                ]
            )
        except LLMError as exc:
            # Advisory layer: a critic outage must not break answering or veto the floor.
            return Validation(
                verdict=Verdict.GROUNDED,
                confidence=0.5,
                detail=f"critic: unavailable ({exc}); deferred to deterministic check.",
            )

        verdict, claims = _interpret(response.content)
        return Validation(
            verdict=verdict,
            confidence=_CONFIDENCE[verdict],
            unsupported_claims=tuple(claims),
            detail=f"critic: {verdict.value}.",
        )


def _build_user_prompt(answer: AgentAnswer, cited: list[Chunk]) -> str:
    sources = "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in cited)
    return (
        f"QUESTION:\n{answer.question}\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"ANSWER:\n{answer.answer}"
    )


def _interpret(raw: str) -> tuple[Verdict, list[str]]:
    """Read a verdict and unsupported claims from the critic's reply, leniently."""
    obj = extract_json_object(raw)
    if obj is not None:
        verdict_word = str(obj.get("verdict", "")).strip().casefold()
        verdict = _VERDICT_WORDS.get(verdict_word)
        if verdict is not None:
            raw_claims = obj.get("unsupported_claims")
            claims = (
                [str(claim) for claim in raw_claims][:_MAX_CLAIMS]
                if isinstance(raw_claims, list)
                else []
            )
            return verdict, claims

    # No parseable JSON verdict: scan for an explicit objection, else raise none.
    low = raw.casefold()
    if any(word in low for word in ("ungrounded", "unsupported", "not supported", "not fully")):
        return Verdict.UNGROUNDED, []
    if "weak" in low or "partial" in low:
        return Verdict.WEAK, []
    return Verdict.GROUNDED, []
