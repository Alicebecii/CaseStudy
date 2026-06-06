"""Small argument-handling helpers shared by the tool modules.

Tool argument handling is deliberately lenient - small local models name arguments slightly
differently ("id" vs "section_id"), so tools accept a few aliases via :func:`first` rather
than failing a whole step on a cosmetic mismatch.
"""

from __future__ import annotations


def first(kwargs: dict, *names: str) -> object:
    """Return the first present keyword among ``names`` (tolerates argument aliases)."""
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


def as_int(value: object, default: int) -> int:
    """Coerce a tool argument to int, accepting ints and numeric strings."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return default
