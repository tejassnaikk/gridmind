"""
Retrieval strategy functions for GridMind eval.

Each strategy shares the signature:
    strategy(question: str, conn: psycopg.Connection, k: int,
             *, prior_column: str = "obligation_strength") -> list[str]

Returning a list of chunk_ids (strings), top-k, in rank order.

prior_column is accepted by every strategy for signature uniformity.
Strategies that do not apply priors (lexical_only, dense_only, rrf_hybrid)
accept but ignore it.  Strategies that call query() or query_with_expansion()
pass it through so the correct obligation column is used.

All seven strategies are collected in STRATEGIES for the runner to iterate.
"""

from __future__ import annotations

import numpy as np
import psycopg

from ingestion.embed import _get_model
from retrieval.config import RETRIEVAL_CONFIG
from retrieval.crossref_expand import query_with_expansion
from retrieval.query import query
from retrieval.retrievers import dense_search, sparse_search
from retrieval.scorer import rrf_fuse

# BGE query instruction prefix — passages are embedded without this.
_BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def _embed(question: str) -> np.ndarray:
    model = _get_model()
    vec: np.ndarray = model.encode(
        _BGE_PREFIX + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype("float32")


# ---------------------------------------------------------------------------
# Five strategies
# ---------------------------------------------------------------------------

def lexical_only(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """Lexical ts_rank retrieval only. No fusion, no priors."""
    return sparse_search(conn, question, k)


def dense_only(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """Dense cosine retrieval only. No fusion, no priors."""
    qvec = _embed(question)
    return dense_search(conn, qvec, k)


def rrf_hybrid(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """
    RRF fusion of dense + lexical. No priors applied.

    Uses candidate_pool from config for both retrievers, then fuses by rank
    and returns top-k chunk ids sorted by RRF score descending.
    """
    cfg = RETRIEVAL_CONFIG
    pool = cfg["candidate_pool"]
    qvec = _embed(question)
    dense_ids = dense_search(conn, qvec, pool)
    sparse_ids = sparse_search(conn, question, pool)
    scores = rrf_fuse(dense_ids, sparse_ids, cfg)
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return sorted_ids[:k]


def rrf_priors(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """RRF + freshness × obligation priors. Production base strategy."""
    results = query(question, k=k, conn=conn, prior_column=prior_column)
    return [r["_chunk_id"] for r in results]


def rrf_priors_crossref(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """RRF + priors + cross-reference expansion. Production expansion strategy."""
    results = query_with_expansion(question, k=k, conn=conn, prior_column=prior_column)
    return [r["_chunk_id"] for r in results]


def rrf_priors_v2(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """RRF + freshness × obligation priors using the v2 (LogReg-derived) column."""
    results = query(question, k=k, conn=conn, prior_column="obligation_strength_v2")
    return [r["_chunk_id"] for r in results]


def rrf_priors_crossref_v2(
    question: str, conn: psycopg.Connection, k: int,
    *, prior_column: str = "obligation_strength",
) -> list[str]:
    """RRF + priors + cross-reference expansion using the v2 (LogReg-derived) column."""
    results = query_with_expansion(question, k=k, conn=conn, prior_column="obligation_strength_v2")
    return [r["_chunk_id"] for r in results]


# ---------------------------------------------------------------------------
# Strategy registry — iterated by eval/run.py
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, callable] = {
    "lexical":                  lexical_only,
    "dense":                    dense_only,
    "rrf":                      rrf_hybrid,
    "rrf+priors":               rrf_priors,
    "rrf+priors+crossref":      rrf_priors_crossref,
    "rrf+priors_v2":            rrf_priors_v2,
    "rrf+priors+crossref_v2":   rrf_priors_crossref_v2,
}
