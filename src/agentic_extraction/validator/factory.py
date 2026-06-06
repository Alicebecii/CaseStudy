"""The seam that selects the validator.

Today it returns the deterministic :class:`GroundingValidator`. The ``llm`` parameter is
the forward-compatible hook for the optional LLM critic layer: when that lands, this is the
one place that composes the deterministic check with a model-based second opinion, so no
caller changes.
"""

from __future__ import annotations

from ..config import Settings
from ..core.interfaces import LLMProvider, Validator
from .grounding import GroundingValidator


def create_validator(settings: Settings, llm: LLMProvider | None = None) -> Validator:
    """Build the validator for the current configuration (deterministic grounding for now)."""
    return GroundingValidator()
