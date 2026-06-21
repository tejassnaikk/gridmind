"""
Reciprocal Rank Fusion and regulatory-prior scoring for GridMind.

Pipeline:
  1. rrf_fuse        — fuse dense + lexical ranked lists by rank (never by raw score)
  2. apply_priors    — multiply by freshness * obligation  (ablatable; keep separate)
  3. rank_candidates — orchestrate both steps, return top-final_k

IMPORTANT: cosine distances and ts_rank values are on different scales and must
NEVER be mixed additively.  RRF operates on list positions only, which removes
the scale mismatch entirely.
"""

from __future__ import annotations


def rrf_fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    cfg: dict,
) -> dict[str, float]:
    """
    Combine two ranked lists with Reciprocal Rank Fusion.

    rrf(c) = w_dense / (rrf_k + rank_dense)  [if present in dense list]
           + w_sparse / (rrf_k + rank_sparse) [if present in sparse list]

    Ranks are 1-based (position 0 in the list = rank 1).
    """
    k = cfg["rrf_k"]
    w_dense = cfg["w_dense"]
    w_sparse = cfg["w_sparse"]

    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_dense / (k + rank)
    for rank, chunk_id in enumerate(sparse_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_sparse / (k + rank)
    return scores


def apply_priors(relevance: float, meta: dict, cfg: dict) -> float:
    """
    Multiply an RRF relevance score by freshness and obligation priors.

    Both priors are MILD nudges — not hard gates.  Keep this function
    independently callable so either prior can be ablated in eval runs.
    """
    fresh = (
        cfg["freshness_current"]
        if meta["is_current"]
        else cfg["freshness_superseded"]
    )
    obl = cfg["obligation"].get(meta["obligation_strength"], 1.0)
    return relevance * fresh * obl


def rank_candidates(
    dense_ids: list[str],
    sparse_ids: list[str],
    meta_by_id: dict[str, dict],
    cfg: dict,
) -> list[tuple[str, float]]:
    """
    Full retrieval scoring pipeline.

    Step 1 (fuse)   — RRF over dense + sparse ranked lists.
    Step 2 (priors) — multiply by freshness * obligation for each candidate.
    Returns the top cfg['final_k'] (id, final_score) pairs, sorted descending.

    Chunks absent from meta_by_id are silently dropped (they have no metadata
    to score against and would never be returned in a valid query path).
    """
    fused = rrf_fuse(dense_ids, sparse_ids, cfg)

    scored: list[tuple[str, float]] = [
        (chunk_id, apply_priors(rrf_score, meta_by_id[chunk_id], cfg))
        for chunk_id, rrf_score in fused.items()
        if chunk_id in meta_by_id
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: cfg["final_k"]]
