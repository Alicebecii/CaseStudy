"""Reciprocal Rank Fusion - the algorithm that merges several ranked lists into one.

This is kept as a pure function, separate from any retriever, for two reasons: it is the
one piece of real ranking logic worth testing in isolation, and it is the seam for the
multimodal extension. RRF takes *N* ranked lists and only looks at *ranks*, so adding a
third list later (e.g. an image-based retriever) fuses in with no change here.

We fuse by rank, not by score, on purpose (DESIGN.md §3.3): BM25 and cosine scores live on
incomparable scales, so normalising them is fragile. A item's contribution from each list
is ``1 / (k + rank)``; ``k`` damps the influence of the very top ranks so a item that
places *well in several lists* can beat one that places *first in only one*.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_id_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking.

    Each input list is ordered best-first. Returns ``(id, fused_score)`` pairs sorted by
    score descending, with ids broken on lexical order so the result is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
