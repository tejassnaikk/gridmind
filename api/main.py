"""
GridMind FastAPI application.

Endpoints:
  POST /query   — retrieval with optional cross-reference expansion
  GET  /health  — liveness probe; exercises the connection pool

Run with:
  uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool

from ingestion.embed import _get_model
from retrieval.query import query
from retrieval.crossref_expand import query_with_expansion
from retrieval.retrievers import register_vector

load_dotenv()


# ---------------------------------------------------------------------------
# Lifespan — pool + embedder warm-up
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ConnectionPool(
        os.environ["DATABASE_URL"],
        min_size=2,
        max_size=10,
        open=True,
    )
    _get_model()            # warm the BGE embedder; first request won't pay ~1 s load
    app.state.pool = pool
    yield {"pool": pool}
    pool.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan, title="GridMind", version="0.1.0")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = None
    expand: bool = False


class QueryResult(BaseModel):
    rank: int
    chunk_id: str
    standard_id: str
    version: int
    requirement_id: str | None
    page_number: int | None
    score: float
    body: str
    from_crossref: bool = False


class QueryResponse(BaseModel):
    question: str
    expand: bool
    count: int
    results: list[QueryResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    with request.app.state.pool.connection() as conn:
        register_vector(conn)
        if body.expand:
            raw = query_with_expansion(body.question, k=body.k, conn=conn)
        else:
            raw = query(body.question, k=body.k, conn=conn)

    results = [
        QueryResult(
            rank=r["rank"],
            chunk_id=r["_chunk_id"],
            standard_id=r["standard_id"],
            version=r["version"],
            requirement_id=r.get("requirement_id"),
            page_number=r.get("page_number"),
            score=r["score"],
            body=r["body"],
            from_crossref=r.get("from_crossref", False),
        )
        for r in raw
    ]

    return QueryResponse(
        question=body.question,
        expand=body.expand,
        count=len(results),
        results=results,
    )


@app.get("/health")
def health(request: Request):
    try:
        with request.app.state.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": str(exc)[:200]},
        )
