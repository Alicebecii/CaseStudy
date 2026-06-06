"""The seam that selects the validator.

The deterministic :class:`GroundingValidator` is always the floor. When a *real* LLM
backend is configured and a provider is supplied, it is paired with an
:class:`LLMCriticValidator` for an independent second opinion (DESIGN.md §3.5). The mock
backend is excluded because a mock critic has nothing useful to say - so tests and the
zero-setup default keep the cheap deterministic check alone.
"""

from __future__ import annotations

from ..config import ProviderName, Settings
from ..core.interfaces import LLMProvider, Validator
from .composite import CompositeValidator
from .critic import LLMCriticValidator
from .grounding import GroundingValidator


def create_validator(settings: Settings, llm: LLMProvider | None = None) -> Validator:
    """Build the validator: deterministic alone, or deterministic + LLM critic."""
    grounding = GroundingValidator()
    if llm is not None and settings.llm_provider is not ProviderName.MOCK:
        return CompositeValidator([grounding, LLMCriticValidator(llm)])
    return grounding
