"""
End-to-end retrieval for GridMind.

Single public entry point: query(question, k) -> list[dict]

Pipeline
--------
1. Embed the question with the BGE query instruction prefix (critical: passages
   are embedded without a prefix; queries MUST use the prefix or cosine
   similarity degrades significantly).
2. Open one psycopg connection, register pgvector.
3. dense_search + sparse_search in parallel SQL queries (same connection).
4. fetch_chunk_meta for the union of candidate IDs.
5. rank_candidates (RRF fuse → freshness * obligation priors) → top-k.
6. Assemble and return result dicts.
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
from dotenv import load_dotenv

from ingestion.embed import _get_model          # shared singleton — no second load
from retrieval.config import RETRIEVAL_CONFIG
from retrieval.retrievers import (
    dense_search,
    fetch_chunk_meta,
    register_vector,
    sparse_search,
)
from retrieval.scorer import rank_candidates

load_dotenv()

# BGE asymmetric embedding: queries use this prefix; passages do not.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _embed_query(question: str) -> np.ndarray:
    model = _get_model()
    text = _QUERY_PREFIX + question
    vec: np.ndarray = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype("float32")


def query(question: str, k: int | None = None) -> list[dict]:
    """
    End-to-end retrieval: embed query → dense + sparse → RRF fuse → priors → top-k.

    Returns a list of dicts with:
        rank, standard_id, version, requirement_id, page_number, score, body.
    """
    cfg = RETRIEVAL_CONFIG
    if k is not None:
        cfg = {**cfg, "final_k": k}     # shallow copy; don't mutate the global

    pool = cfg["candidate_pool"]

    # Step 1: embed
    qvec = _embed_query(question)

    # Steps 2–3: retrieve candidates
    database_url = os.environ["DATABASE_URL"]
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        dense_ids = dense_search(conn, qvec, pool)
        sparse_ids = sparse_search(conn, question, pool)

        # Step 4: fetch metadata for the union of both candidate sets
        all_ids = list(dict.fromkeys(dense_ids + sparse_ids))   # preserve order, dedup
        meta_by_id = fetch_chunk_meta(conn, all_ids)

    # Step 5: RRF fuse + priors → top-k
    ranked = rank_candidates(dense_ids, sparse_ids, meta_by_id, cfg)

    # Step 6: assemble output
    results: list[dict] = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        m = meta_by_id[chunk_id]
        results.append(
            {
                "rank": rank,
                "standard_id": m["standard_id"],
                "version": m["version"],
                "requirement_id": m["requirement_id"],
                "page_number": m["page_number"],
                "score": score,
                "body": m["body"],
            }
        )

    return results
