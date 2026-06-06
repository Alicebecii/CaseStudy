"""Combine several validators into one verdict.

Like :class:`HybridRetriever`, this composes - it holds sub-validators and merges their
results, with no judging logic of its own. The combination rule is deliberately
conservative (DESIGN.md §3.5: refuse to over-claim): the *worst* verdict wins, so the
deterministic check stays a floor that the LLM critic can lower or add to, never raise.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.interfaces import Validator
from ..core.models import AgentAnswer, Chunk, Validation, Verdict

_RANK = {Verdict.GROUNDED: 2, Verdict.WEAK: 1, Verdict.UNGROUNDED: 0}


class CompositeValidator(Validator):
    """Run several validators and combine them: worst verdict, unioned claims."""

    def __init__(self, validators: Sequence[Validator]) -> None:
        if not validators:
            raise ValueError("CompositeValidator requires at least one validator")
        self._validators = tuple(validators)

    def validate(self, answer: AgentAnswer, chunks_by_id: dict[str, Chunk]) -> Validation:
        results = [validator.validate(answer, chunks_by_id) for validator in self._validators]

        worst = min(results, key=lambda result: _RANK[result.verdict])
        confidence = min(result.confidence for result in results)

        claims: dict[str, None] = {}
        for result in results:
            for claim in result.unsupported_claims:
                claims.setdefault(claim, None)

        detail = " | ".join(result.detail for result in results if result.detail)

        return Validation(
            verdict=worst.verdict,
            confidence=round(confidence, 3),
            unsupported_claims=tuple(claims),
            detail=detail,
        )
