# GridMind — Build Context

Energy compliance RAG over NERC/FERC/NREL standards. Plain-English question in;
cited + version-aware + obligation-tagged answer out.

## Locked decisions (do not redesign)
- Single idempotent Python ingestion job. NO Step Functions, EventBridge, DynamoDB, Lambda fan-out.
- Idempotency: SHA256 of source PDF stored in Postgres metadata; skip a doc if hash unchanged and status=complete.
- Vector store: Postgres + pgvector. Keyword: Postgres full-text (ts_rank).
- Embeddings: BAAI/bge-small-en-v1.5 (384-dim, 512-token window). Benchmark one stronger model during the eval phase.

## Retrieval (the differentiator — get this exactly right)
Two independent retrievers, each returns a ranked list of chunk ids:
  dense:  pgvector cosine (embedding <=> qvec), top-40
  sparse: ts_rank over body_tsvector, top-40
Fuse by RANK, not raw score, with Reciprocal Rank Fusion:
  rrf(c) = w_dense/(k+rank_dense) + w_sparse/(k+rank_sparse),  k=60
Then apply regulatory priors as MULTIPLIERS (separate, ablatable stage):
  final(c) = rrf(c) * freshness(c) * obligation(c)
NEVER additively mix cosine and ts_rank — different scales, that is the bug we are avoiding.

## Honesty rules (interview-defensible commitments)
- ts_rank is lexical ranking, NOT BM25. Label it "lexical" everywhere unless pg_search/ParadeDB is installed.
- Obligation prior is a MILD nudge (shall 1.0, should 0.92, may 0.85, informational 0.80), not a cliff.
  It can hurt definitional queries, so keep it separately ablatable.
- Tune weights (w_sparse, rrf_k, obligation factors) on a DEV split; report on a HELD-OUT test split.

## Eval
- One gold chunk per question -> metrics are Recall@5 and MRR. No nDCG unless graded/multi-relevant labels.
- Eval set is partly LLM-generated from held-out chunks (circular bias); hand-verify a hard
  cross-reference subset of ~30-50 questions.
- Strategies: lexical-only / dense-only / RRF hybrid / +priors / +crossref-expansion.

## Build order
1. db/schema.sql            metadata + chunks tables, pgvector + tsvector + indexes
2. retrieval/config.py      RETRIEVAL_CONFIG
3. retrieval/retrievers.py  dense + sparse SQL retrievers
4. retrieval/scorer.py      rrf_fuse + apply_priors
5. retrieval/crossref_expand.py
6. api/main.py              FastAPI POST /query
7. ingestion/*              fetch -> chunk -> classify -> crossref -> embed -> upsert
8. classifier/train.py      DistilBERT obligation classifier (week 3)
9. eval/*                   generate_qa + benchmark
10. README benchmark table

## Stack
Python 3.11. Postgres 15+ with pgvector. FastAPI. Streamlit UI. Start with NERC CIP family (12 standards).
