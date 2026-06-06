"""Cross-run memory (DESIGN.md §3.6).

An explicit, inspectable store of past answered questions, keyed by document. Behind the
:class:`MemoryStore` contract so a different backend can replace it. The file-based
:class:`JsonMemoryStore` is the MVP implementation; wiring recall into the agent loop is
the documented cross-task-learning extension.
"""

from __future__ import annotations

from .store import JsonMemoryStore

__all__ = ["JsonMemoryStore"]
