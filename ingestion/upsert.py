"""
Idempotent document upsert for GridMind ingestion.

Idempotency contract:
  - If (standard_id, version) already exists with the same source_hash
    AND status='complete', skip silently.
  - Otherwise: upsert metadata (status='pending'), delete existing chunks
    for that document_id, insert all fresh chunks, wire prev/next links,
    set status='complete'.  All writes are one atomic transaction.

Prev/next FK notes:
  - standard_chunks.previous_chunk_id / next_chunk_id are DEFERRABLE
    INITIALLY DEFERRED, so the entire chunk list can be inserted with
    forward references in a single transaction — the FK check fires at
    COMMIT, not per-statement.
  - UUIDs are generated client-side so prev/next are known before INSERT.
"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def upsert_document(
    conn: psycopg.Connection,
    meta: dict,
    chunks: list[dict],
) -> None:
    """
    Persist one standard document and its chunks.

    *meta* keys expected:
      standard_id, version, standard_family, title, source_hash, file_path,
      total_chunks (ignored — recomputed from chunks),
      effective_date (may be None), superseded_by (may be None),
      is_external (default False).

    *chunks* must have been through embed_chunks (embedding present) and
    classify/crossref (obligation_strength, related_standards present).
    """
    register_vector(conn)

    standard_id: str = meta["standard_id"]
    version: int = meta["version"]
    source_hash: str = meta["source_hash"]

    # ------------------------------------------------------------------ #
    # Idempotency check (read-only, outside the write transaction)
    # ------------------------------------------------------------------ #
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_hash, status
            FROM   standard_document_metadata
            WHERE  standard_id = %s AND version = %s
            """,
            [standard_id, version],
        )
        row = cur.fetchone()

    if row and row[0] == source_hash and row[1] == "complete":
        print(f"{standard_id} v{version}: unchanged, skipping")
        return

    # ------------------------------------------------------------------ #
    # Write transaction
    # ------------------------------------------------------------------ #
    with conn.transaction():
        with conn.cursor() as cur:

            # -- Upsert metadata row, get document_id --
            cur.execute(
                """
                INSERT INTO standard_document_metadata
                  (standard_id, version, standard_family, title,
                   effective_date, superseded_by, is_external,
                   source_hash, status, file_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                ON CONFLICT (standard_id, version) DO UPDATE SET
                  source_hash  = EXCLUDED.source_hash,
                  title        = EXCLUDED.title,
                  file_path    = EXCLUDED.file_path,
                  status       = 'pending',
                  updated_at   = NOW()
                RETURNING document_id
                """,
                [
                    standard_id,
                    version,
                    meta["standard_family"],
                    meta.get("title"),
                    meta.get("effective_date"),
                    meta.get("superseded_by"),
                    meta.get("is_external", False),
                    source_hash,
                    meta["file_path"],
                ],
            )
            document_id: uuid.UUID = cur.fetchone()[0]

            # -- Delete existing chunks (safe re-ingestion on hash change) --
            cur.execute(
                "DELETE FROM standard_chunks WHERE document_id = %s",
                [document_id],
            )

            # -- Assign UUIDs client-side so prev/next are known at INSERT --
            chunk_ids = [uuid.uuid4() for _ in chunks]
            n = len(chunks)

            for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                prev_id = chunk_ids[i - 1] if i > 0 else None
                next_id = chunk_ids[i + 1] if i < n - 1 else None

                # Drop self-references; keep cross-document citations only.
                raw_refs = chunk.get("related_standards") or []
                refs = [r for r in raw_refs if r != standard_id] or None

                emb = chunk.get("embedding")
                if emb is not None and not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype="float32")

                cur.execute(
                    """
                    INSERT INTO standard_chunks
                      (id, document_id, chunk_index, requirement_id,
                       obligation_strength, body, augmented_text, embedding,
                       related_standards, page_number,
                       previous_chunk_id, next_chunk_id)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        chunk_id,
                        document_id,
                        chunk["chunk_index"],
                        chunk.get("requirement_id"),
                        chunk.get("obligation_strength"),
                        chunk["body"],
                        chunk.get("augmented_text"),
                        emb,
                        refs,
                        chunk.get("page_number"),
                        prev_id,
                        next_id,
                    ],
                )

            # -- Mark complete --
            cur.execute(
                """
                UPDATE standard_document_metadata
                SET    status       = 'complete',
                       total_chunks = %s,
                       updated_at   = NOW()
                WHERE  document_id  = %s
                """,
                [n, document_id],
            )

    print(f"{standard_id} v{version}: {n} chunks ingested (document_id={document_id})")
