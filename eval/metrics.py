"""
Standard IR metrics. Used by eval/run.py.
"""


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Fraction of gold chunks present in the top-k retrieved.
    With single-gold sets this collapses to {0.0, 1.0}; with multi-gold
    it lives in [0, 1]. Returns 0.0 if gold_ids is empty.
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for g in gold_ids if g in top_k)
    return hits / len(gold_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    1 / (rank of the first gold chunk found in retrieved_ids), 1-indexed.
    Returns 0.0 if no gold chunk is in the retrieved list.
    """
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0
