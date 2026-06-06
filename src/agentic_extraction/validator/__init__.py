"""Validation: check that an answer is grounded in the chunks it cites.

The deterministic :class:`GroundingValidator` (no LLM) is the mandatory check;
:func:`answer_with_validation` wraps the agent with it and retries once on a failing
verdict (DESIGN.md §3.5). Use :func:`create_validator` to obtain the configured validator.
"""

from __future__ import annotations

from .factory import create_validator
from .grounding import GroundingValidator
from .pipeline import answer_with_validation

__all__ = [
    "create_validator",
    "GroundingValidator",
    "answer_with_validation",
]
