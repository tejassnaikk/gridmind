"""
Cross-reference expansion retrieval strategy for GridMind.

Public entry point: query_with_expansion(question, k) -> list[dict]

This strategy runs the standard base pipeline first, then re-queries the
database scoped to each standard that the top-k results cite.  Expansion
chunks are penalised by RETRIEVAL_CONFIG['crossref_penalty'] (0.85) so
they only displace a base result when they are meaningfully more relevant.

Intended use:
  - Benchmarking: compare Recall@k / MRR of base vs. +crossref strategies.
  - Keep this file self-contained; do NOT modify retrievers.py, scorer.py,
    config.py, or query.py — benchmark integrity depends on those being fixed.
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
from dotenv import load_dotenv

from ingestion.embed import _get_model           # shared singleton — no second load
from retrieval.config import RETRIEVAL_CONFIG
from retrieval.retrievers import (
    dense_search,
    fetch_chunk_meta,
    register_vector,
    sparse_search,
)
from retrieval.scorer import apply_priors, rank_candidates, rrf_fuse

load_dotenv()

# BGE asymmetric embedding — identical to the constant in retrieval/query.py.
# Defined locally to avoid importing a private name across sibling modules.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Candidate pool for scoped cross-ref sub-queries.  Smaller than the base
# pool (40) because we are already scoped to one standard.
_CROSSREF_POOL = 10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _embed_query(question: str) -> np.ndarray:
    model = _get_model()
    vec: np.ndarray = model.encode(
        _QUERY_PREFIX + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype("float32")


def _dense_search_scoped(
    conn: psycopg.Connection,
    query_vector: np.ndarray,
    pool: int,
    standard_id: str,
) -> list[str]:
    """dense_search filtered to one standard_id — local variant, retrievers.py unchanged."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sc.id::text
            FROM   standard_chunks sc
            WHERE  sc.document_id IN (
                       SELECT document_id
                       FROM   standard_document_metadata
                       WHERE  standard_id = %s
                   )
            ORDER  BY sc.embedding <=> %s::vector
            LIMIT  %s
            """,
            [standard_id, query_vector, pool],
        )
        return [row[0] for row in cur.fetchall()]


def _sparse_search_scoped(
    conn: psycopg.Connection,
    query_text: str,
    pool: int,
    standard_id: str,
) -> list[str]:
    """sparse_search (lexical ts_rank, NOT BM25) filtered to one standard_id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sc.id::text
            FROM   standard_chunks sc,
                   websearch_to_tsquery('english', %s) AS q
            WHERE  sc.document_id IN (
                       SELECT document_id
                       FROM   standard_document_metadata
                       WHERE  standard_id = %s
                   )
              AND  sc.body_tsvector @@ q
            ORDER  BY ts_rank(sc.body_tsvector, q) DESC
            LIMIT  %s
            """,
            [query_text, standard_id, pool],
        )
        return [row[0] for row in cur.fetchall()]


def _fetch_related_standards(
    conn: psycopg.Connection,
    ids: list[str],
) -> dict[str, list[str]]:
    """
    Return {chunk_id: [standard_id, ...]} for the given chunk ids.

    fetch_chunk_meta does not return related_standards, so this is a
    separate targeted query used only for the expansion step.
    """
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, related_standards
            FROM   standard_chunks
            WHERE  id::text = ANY(%s)
              AND  related_standards IS NOT NULL
            """,
            [ids],
        )
        return {
            row[0]: row[1]
            for row in cur.fetchall()
            if row[1]
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def query_with_expansion(
    question: str,
    k: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[dict]:
    """
    Base pipeline + cross-reference expansion.

    Same return shape as retrieval.query.query() with one extra key per dict:
      'from_crossref' (bool) — True if this chunk was surfaced via expansion.

    Behavior notes (verified via instrumented runs):
      - Cross-references are collected from the FULL base candidate pool
        (top ~40), not the displayed top-K. Citation-bearing chunks
        (applicability, guidelines) almost never outrank requirement bodies,
        so a top-K-only source would silently disable expansion in practice.
      - Expansion only retrieves from referenced standards that are actually
        ingested. If CIP-013-2 cites CIP-003 but CIP-003 isn't in the DB,
        that branch returns zero candidates and is skipped.
      - Expansion candidates carry a `crossref_penalty` (0.85) on top of
        their RRF+priors score. When a referenced standard is *also*
        retrieved by base, the base entries usually outrank the expansion
        entries — by design. Expansion adds depth on queries where base
        misses the referenced standard entirely.
    """
    cfg = RETRIEVAL_CONFIG
    final_k = k if k is not None else cfg["final_k"]
    base_cfg = {**cfg, "final_k": final_k}

    # Step 1: embed the query once
    qvec = _embed_query(question)

    # Steps 2–4: all DB work in one connection
    _owned = conn is None
    if _owned:
        conn = psycopg.connect(os.environ["DATABASE_URL"])
        register_vector(conn)
    try:
        # ------------------------------------------------------------------
        # Step 2: base retrieval (identical to retrieval.query.query)
        # ------------------------------------------------------------------
        pool = cfg["candidate_pool"]
        dense_ids = dense_search(conn, qvec, pool)
        sparse_ids = sparse_search(conn, question, pool)

        all_base_ids = list(dict.fromkeys(dense_ids + sparse_ids))
        meta_by_id = fetch_chunk_meta(conn, all_base_ids)

        ranked_base = rank_candidates(dense_ids, sparse_ids, meta_by_id, base_cfg)

        base_results: list[dict] = []
        for chunk_id, score in ranked_base:
            m = meta_by_id[chunk_id]
            base_results.append(
                {
                    "standard_id": m["standard_id"],
                    "version": m["version"],
                    "requirement_id": m["requirement_id"],
                    "page_number": m["page_number"],
                    "score": score,
                    "body": m["body"],
                    "from_crossref": False,
                    "_chunk_id": chunk_id,
                }
            )

        # ------------------------------------------------------------------
        # Step 3: collect cross-ref standards from the FULL candidate pool,
        # not just the top-K displayed base results.
        #
        # Rationale: in regulatory standards, citation-bearing chunks
        # (applicability sections, introductions, guidelines) almost never
        # outrank the requirement bodies that actually answer the query.
        # If we only look at the top-K, refs_by_chunk is empty on virtually
        # every real query, and expansion silently never fires. Looking at
        # the full candidate pool lets us discover the cross-ref graph
        # regardless of which chunks ranked high on this particular query.
        # ------------------------------------------------------------------
        candidate_pool_ids = list(dict.fromkeys(dense_ids + sparse_ids))
        refs_by_chunk = _fetch_related_standards(conn, candidate_pool_ids)
        base_chunk_ids = [r["_chunk_id"] for r in base_results]

        # Collect every standard referenced by any base-result chunk.
        # We deliberately do NOT subtract standards already present in
        # base_results: in regulatory corpora the cited standards are
        # exactly the ones that also surface in dense retrieval, so
        # presence-based suppression silently disables expansion on the
        # most common case. Redundancy is handled later by chunk-id
        # dedupe in the merge step.
        # Iterate over the FULL candidate pool, not just top-K base results.
        # refs_by_chunk was widened above to cover candidate_pool_ids; we must
        # consume it over the same wider key set, otherwise the widening has
        # no effect (we'd only ever look up the 5 keys that match base).
        xref_standards: set[str] = set()
        for chunk_id in candidate_pool_ids:
            for std in refs_by_chunk.get(chunk_id, []):
                xref_standards.add(std)

        # ------------------------------------------------------------------
        # Step 4: expand into each referenced standard
        # ------------------------------------------------------------------
        base_chunk_id_set = {r["_chunk_id"] for r in base_results}
        xref_cfg = {**cfg, "final_k": 3}   # top 3 per referenced standard

        xref_results: list[dict] = []

        for xref_std in sorted(xref_standards):   # sorted for determinism
            d_ids = _dense_search_scoped(conn, qvec, _CROSSREF_POOL, xref_std)
            s_ids = _sparse_search_scoped(conn, question, _CROSSREF_POOL, xref_std)

            if not d_ids and not s_ids:
                continue

            xref_candidate_ids = list(dict.fromkeys(d_ids + s_ids))
            xref_meta = fetch_chunk_meta(conn, xref_candidate_ids)

            fused = rrf_fuse(d_ids, s_ids, cfg)

            xref_scored: list[tuple[str, float]] = [
                (
                    cid,
                    apply_priors(score, xref_meta[cid], cfg) * cfg["crossref_penalty"],
                )
                for cid, score in fused.items()
                if cid in xref_meta
            ]
            xref_scored.sort(key=lambda x: x[1], reverse=True)

            for cid, score in xref_scored[:3]:
                if cid in base_chunk_id_set:
                    continue   # already surfaced by base path — keep base entry
                m = xref_meta[cid]
                xref_results.append(
                    {
                        "standard_id": m["standard_id"],
                        "version": m["version"],
                        "requirement_id": m["requirement_id"],
                        "page_number": m["page_number"],
                        "score": score,
                        "body": m["body"],
                        "from_crossref": True,
                        "_chunk_id": cid,
                    }
                )

    finally:
        if _owned:
            conn.close()

    # ----------------------------------------------------------------------
    # Step 5: merge, dedup (base beats xref for same chunk), sort, re-rank
    # ----------------------------------------------------------------------
    seen_ids: set[str] = set(base_chunk_id_set)
    merged = list(base_results)
    for r in xref_results:
        if r["_chunk_id"] not in seen_ids:
            merged.append(r)
            seen_ids.add(r["_chunk_id"])

    merged.sort(key=lambda r: r["score"], reverse=True)
    merged = merged[:final_k]

    results: list[dict] = []
    for rank, r in enumerate(merged, start=1):
        results.append(
            {
                "rank": rank,
                "_chunk_id": r["_chunk_id"],
                "standard_id": r["standard_id"],
                "version": r["version"],
                "requirement_id": r["requirement_id"],
                "page_number": r["page_number"],
                "score": r["score"],
                "body": r["body"],
                "from_crossref": r["from_crossref"],
            }
        )

    return results
