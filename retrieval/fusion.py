from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def rrf_fusion(
    dense_ranks: Sequence[Tuple[int, int]],
    sparse_ranks: Sequence[Tuple[int, int]],
    k: int = 60,
) -> List[int]:
    """Reciprocal Rank Fusion between dense and sparse rankings.

    Args:
        dense_ranks: List of (doc_id, rank) from dense search (rank 0-indexed).
        sparse_ranks: List of (doc_id, rank) from sparse search (rank 0-indexed).
        k: RRF constant (default 60).

    Returns:
        List of doc_ids sorted by descending RRF score.
    """
    scores: Dict[int, float] = {}

    for doc_id, rank in dense_ranks:
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for doc_id, rank in sparse_ranks:
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]
