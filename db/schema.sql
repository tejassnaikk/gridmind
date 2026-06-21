-- GridMind schema
-- Requires Postgres 15+ with pgvector extension.
-- Run once: CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Table 1: standard_document_metadata
-- One row per (standard_id, version) pair.
-- Idempotency check: skip ingestion when source_hash unchanged AND status='complete'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS standard_document_metadata (
    document_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    standard_id     TEXT        NOT NULL,                   -- e.g. 'CIP-013'
    version         INTEGER     NOT NULL,
    standard_family TEXT        NOT NULL,                   -- e.g. 'CIP', 'FAC', 'FERC'
    title           TEXT,
    effective_date  DATE,
    superseded_by   TEXT,                                   -- NULL means this is the current revision
    is_external     BOOLEAN     NOT NULL DEFAULT FALSE,     -- TRUE for FERC orders, NREL reports, etc.
    source_hash     TEXT,                                   -- SHA-256 of source PDF
    status          TEXT,                                   -- 'pending' | 'complete' | 'error'
    file_path       TEXT,
    total_chunks    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_standard_version UNIQUE (standard_id, version)
);

-- ---------------------------------------------------------------------------
-- Table 2: standard_chunks
-- One row per text chunk.  body_tsvector is auto-maintained by Postgres.
-- prev/next links allow context-window expansion at query time.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS standard_chunks (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID        NOT NULL
                            REFERENCES standard_document_metadata (document_id)
                            ON DELETE CASCADE,
    chunk_index         INTEGER     NOT NULL,

    requirement_id      TEXT,                               -- e.g. 'R4.2'; NULL for preamble chunks
    -- Obligation strength drives mild score multipliers during retrieval:
    --   shall=1.0  should=0.92  may=0.85  informational=0.80
    -- Kept separately ablatable; never hard-gates retrieval.
    obligation_strength TEXT        CHECK (
                            obligation_strength IN ('shall', 'should', 'may', 'informational')
                        ),

    body                TEXT        NOT NULL,
    -- augmented_text = body + injected metadata context; this is what gets embedded,
    -- not body, so dense retrieval benefits from structured context without polluting
    -- the lexical index (body_tsvector is generated from body only).
    augmented_text      TEXT,

    -- 384-dim vectors for all-MiniLM-L6-v2.  NULL until embedding job runs.
    embedding           vector(384),

    -- Generated column: always in sync with body, never manually set.
    body_tsvector       tsvector    GENERATED ALWAYS AS (
                            to_tsvector('english', body)
                        ) STORED,

    related_standards   TEXT[],                             -- cross-refs extracted from body, e.g. '{CIP-003,CIP-005}'
    page_number         INTEGER,
    metadata            JSONB,

    -- Doubly-linked list for sliding-window context expansion.
    -- DEFERRABLE so bulk inserts can set both directions in one transaction.
    previous_chunk_id   UUID        REFERENCES standard_chunks (id)
                            DEFERRABLE INITIALLY DEFERRED,
    next_chunk_id       UUID        REFERENCES standard_chunks (id)
                            DEFERRABLE INITIALLY DEFERRED,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Dense retrieval: HNSW for approximate cosine search (pgvector ≥0.5).
-- m=16 ef_construction=64 are safe defaults; tune if recall degrades.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON standard_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lexical retrieval: ts_rank over body_tsvector.
CREATE INDEX IF NOT EXISTS idx_chunks_body_tsvector_gin
    ON standard_chunks
    USING gin (body_tsvector);

-- Chunk navigation and ordered retrieval within a document.
CREATE INDEX IF NOT EXISTS idx_chunks_document_chunk
    ON standard_chunks (document_id, chunk_index);

-- Obligation-strength filtering / ablation experiments.
CREATE INDEX IF NOT EXISTS idx_chunks_obligation_strength
    ON standard_chunks (obligation_strength);

-- Document lookup by (standard_id, version).
CREATE INDEX IF NOT EXISTS idx_metadata_standard_version
    ON standard_document_metadata (standard_id, version);

-- Fast current-revision queries: WHERE superseded_by IS NULL.
CREATE INDEX IF NOT EXISTS idx_metadata_current_revision
    ON standard_document_metadata (standard_id)
    WHERE superseded_by IS NULL;

-- Idempotency check: look up a document by its source PDF hash.
CREATE INDEX IF NOT EXISTS idx_metadata_source_hash
    ON standard_document_metadata (source_hash);
