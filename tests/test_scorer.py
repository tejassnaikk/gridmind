"""
Unit tests for retrieval/scorer.py — no database required.

Two invariants verified:
  (a) A chunk ranked highly by BOTH retrievers beats one ranked highly by only one.
  (b) At close retrieval relevance, a 'shall' current chunk outranks
      a 'may' superseded chunk once priors are applied.
"""

from retrieval.config import RETRIEVAL_CONFIG
from retrieval.scorer import apply_priors, rank_candidates, rrf_fuse


def _meta(obligation: str, is_current: bool) -> dict:
    return {
        "obligation_strength": obligation,
        "is_current": is_current,
        "body": "",
        "requirement_id": None,
        "standard_id": "CIP-013",
        "version": 3,
        "page_number": 1,
    }


# ---------------------------------------------------------------------------
# (a) Dual-retriever rank beats single-retriever rank
# ---------------------------------------------------------------------------

def test_dual_ranked_beats_single_dense():
    """
    'both_high' appears at rank 1 in dense AND rank 1 in sparse.
    'dense_only' appears at rank 2 in dense and nowhere in sparse.
    RRF must score both_high higher without any prior adjustment.
    """
    cfg = RETRIEVAL_CONFIG
    dense_ids = ["both_high", "dense_only", "c3"]
    sparse_ids = ["both_high", "s2", "s3"]

    scores = rrf_fuse(dense_ids, sparse_ids, cfg)

    assert "both_high" in scores
    assert "dense_only" in scores
    assert scores["both_high"] > scores["dense_only"], (
        f"Expected both_high ({scores['both_high']:.6f}) > "
        f"dense_only ({scores['dense_only']:.6f})"
    )


# ---------------------------------------------------------------------------
# (b) 'shall' current outranks 'may' superseded when relevance is close
# ---------------------------------------------------------------------------

def test_shall_current_beats_may_superseded_via_priors():
    """
    'may_sup' ranks 1st in dense (higher raw retrieval score).
    'shall_cur' ranks 2nd in dense.
    After freshness * obligation priors, shall_cur must finish first.

    Numeric check:
      may_sup  RRF = 1.0/(60+1) = 0.01639  → final = 0.01639 * 0.50 * 0.85 = 0.00697
      shall_cur RRF = 1.0/(60+2) = 0.01613  → final = 0.01613 * 1.00 * 1.00 = 0.01613
    """
    cfg = RETRIEVAL_CONFIG
    dense_ids = ["may_sup", "shall_cur"]
    sparse_ids: list[str] = []

    meta = {
        "may_sup": _meta("may", is_current=False),
        "shall_cur": _meta("shall", is_current=True),
    }

    ranked = rank_candidates(dense_ids, sparse_ids, meta, cfg)
    ids_in_order = [chunk_id for chunk_id, _ in ranked]

    assert ids_in_order[0] == "shall_cur", (
        f"Expected shall_cur first, got: {ids_in_order}"
    )


def test_apply_priors_values():
    """Spot-check the four obligation multipliers and freshness factors."""
    cfg = RETRIEVAL_CONFIG
    relevance = 1.0

    assert apply_priors(relevance, _meta("shall", True), cfg) == 1.00 * 1.00
    assert apply_priors(relevance, _meta("should", True), cfg) == 0.92 * 1.00
    assert apply_priors(relevance, _meta("may", True), cfg) == 0.85 * 1.00
    assert apply_priors(relevance, _meta("informational", True), cfg) == 0.80 * 1.00

    assert apply_priors(relevance, _meta("shall", False), cfg) == 1.00 * 0.50
    assert apply_priors(relevance, _meta("may", False), cfg) == 0.85 * 0.50
