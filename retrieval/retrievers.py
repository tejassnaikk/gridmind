"""
Dense (pgvector cosine) and lexical (ts_rank) retrievers for GridMind.

Callers must invoke pgvector.psycopg.register_vector(conn) once on each
connection before calling dense_search so that vector parameters are
serialised correctly by psycopg.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector  # noqa: F401 — re-exported for callers


def dense_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    pool: int,
) -> list[str]:
    """Return up to *pool* chunk ids ordered by cosine distance (closest first)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text
            FROM   standard_chunks
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
            """,
            [query_vector, pool],
        )
        return [row[0] for row in cur.fetchall()]


def sparse_search(
    conn: psycopg.Connection,
    query_text: str,
    pool: int,
) -> list[str]:
    """Return up to *pool* chunk ids ordered by ts_rank (lexical, NOT BM25)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sc.id::text
            FROM   standard_chunks sc,
                   websearch_to_tsquery('english', %s) AS q
            WHERE  sc.body_tsvector @@ q
            ORDER  BY ts_rank(sc.body_tsvector, q) DESC
            LIMIT  %s
            """,
            [query_text, pool],
        )
        return [row[0] for row in cur.fetchall()]


def fetch_chunk_meta(
    conn: psycopg.Connection,
    ids: list[str],
) -> dict[str, dict]:
    """
    Return a mapping of chunk id -> metadata dict for every id in *ids*.

    Joins standard_chunks to standard_document_metadata.
    is_current is True when the parent document has not been superseded
    (superseded_by IS NULL).
    """
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sc.id::text,
                   sc.body,
                   sc.requirement_id,
                   sc.obligation_strength,
                   m.standard_id,
                   m.version,
                   sc.page_number,
                   (m.superseded_by IS NULL) AS is_current
            FROM   standard_chunks sc
            JOIN   standard_document_metadata m
                     ON sc.document_id = m.document_id
            WHERE  sc.id::text = ANY(%s)
            """,
            [ids],
        )
        return {
            row[0]: {
                "body": row[1],
                "requirement_id": row[2],
                "obligation_strength": row[3],
                "standard_id": row[4],
                "version": row[5],
                "page_number": row[6],
                "is_current": row[7],
            }
            for row in cur.fetchall()
        }
