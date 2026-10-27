"""Retrieval metrics — the formulas, written out plainly.

All four functions take ONE query's:
    retrieved : ranked doc ids, best first (chunks already collapsed to docs)
    expected  : gold doc ids (the answer key / qrels)
    k         : how deep down the list to look

Run `python bench/eval_metrics.py` for a hand-checkable worked example.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """recall@k = (# gold docs found in top-k) / (total # gold docs)

    Order-blind: same doc at rank 1 or rank k scores identically.
    Measures coverage; the number that says "did we fetch everything".
    """
    if not expected:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in expected)
    return hits / len(expected)


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """precision@k = (# gold docs in top-k) / k

    Denominator is always k (the display slots you bought), so junk that
    consumes a slot is punished even when recall rises with k.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in expected)
    return hits / k


def mrr_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """MRR@k = 1 / rank_of_first_gold_hit     (0 if no gold in top-k)

    Only the FIRST hit counts; every later gold doc is invisible.
    Linear discount: rank 3 -> 0.33.
    """
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k        ("list quality" / "best possible list")

    DCG@k  = sum over ranks i=1..k of rel_i / log2(i + 1)
             rel_i = 1 if retrieved[i] is gold (binary relevance)
    IDCG@k = same shape with all gold docs packed at the top:
             sum of 1/log2(rank+1) for rank = 1..min(len(expected), k)

    Log discount: rank 2 -> 0.63, rank 3 -> 0.50, rank 4 -> 0.43
    (gentler than MRR's linear 1/rank). Returns 1.0 only when the gold docs
    occupy the very top of the list.
    For graded qrels (rel 0..3) use rel = 2**grade - 1 in the DCG sum.
    """
    if not expected or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in expected
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


if __name__ == "__main__":
    # Lecture example:  gold = {A, B},  ranked list = [X, A, Y, B, Z]
    retrieved = ["X", "A", "Y", "B", "Z"]
    expected = {"A", "B"}

    checks = [
        ("recall@2",     recall_at_k,    2, 0.5),
        ("recall@5",     recall_at_k,    5, 1.0),
        ("precision@2",  precision_at_k, 2, 0.5),
        ("precision@5",  precision_at_k, 5, 0.4),
        ("mrr@5",        mrr_at_k,       5, 0.5),
        ("ndcg@5",       ndcg_at_k,      5, 0.651),
    ]
    for name, fn, k, want in checks:
        got = fn(retrieved, expected, k)
        assert abs(got - want) < 1e-3, f"{name}: got {got:.4f}, want {want}"
        print(f"{name:>12} = {got:.4f}   (expected {want})")
    print("all values match the hand-worked example")
