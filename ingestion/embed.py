"""
Embedding stage for GridMind ingestion.

Model: BAAI/bge-small-en-v1.5 (384-dim, 512-token window).
BGE passage embeddings need NO instruction prefix — queries use a prefix,
passages are embedded as-is.  normalize_embeddings=True so cosine distance
reduces to dot product and pgvector's <=> operator works correctly.
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def build_augmented_text(chunk: dict, standard_id: str) -> str:
    """
    Prepend structured metadata context to the chunk body.

    This is what gets embedded — body_tsvector is generated from body only,
    so augmented_text enriches dense retrieval without polluting the lexical
    index.
    """
    req = chunk.get("requirement_id") or "preamble"
    oblig = chunk.get("obligation_strength") or ""
    refs = chunk.get("related_standards") or []

    prefix = f"{standard_id} {req}"
    if oblig:
        prefix += f" [{oblig}]"
    if refs:
        prefix += f" related: {', '.join(refs)}"

    return f"{prefix}\n\n{chunk['body']}"


def embed_chunks(chunks: list[dict], standard_id: str) -> list[dict]:
    """
    Add 'augmented_text' (str) and 'embedding' (np.ndarray float32, 384-dim)
    to each chunk dict in-place.  Returns the same list.
    """
    model = _get_model()

    texts: list[str] = []
    for chunk in chunks:
        aug = build_augmented_text(chunk, standard_id)
        chunk["augmented_text"] = aug
        texts.append(aug)

    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.astype("float32")

    return chunks
