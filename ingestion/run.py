"""
End-to-end ingestion runner.

Usage:
    python -m ingestion.run data/raw/CIP-013-2.pdf

Pipeline:
    parse -> chunk -> classify -> crossref -> embed -> upsert

Filename convention expected:  <FAMILY>-<NNN>-<VERSION>.pdf
    CIP-013-2.pdf  ->  standard_id=CIP-013  version=2  standard_family=CIP
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from ingestion.chunk import chunk_standard
from ingestion.classify import classify_obligation
from ingestion.crossref import extract_related_standards
from ingestion.embed import embed_chunks
from ingestion.parse import extract_pages
from ingestion.upsert import file_sha256, upsert_document

load_dotenv()

# e.g. "CIP-013-2" -> family=CIP, number=013, version=2
_FILENAME_RE = re.compile(r"^([A-Z]{2,3})-(\d{3})-(\d+)$", re.IGNORECASE)

# "Title: Cyber Security - Supply Chain Risk Management"
_TITLE_RE = re.compile(r"Title:\s*(.+)", re.IGNORECASE)


def _parse_filename(stem: str) -> tuple[str, str, int]:
    """Return (standard_id, standard_family, version) from a bare filename stem."""
    m = _FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Filename '{stem}' does not match expected pattern "
            f"<FAMILY>-<NNN>-<VERSION> (e.g. CIP-013-2)"
        )
    family, number, version = m.group(1).upper(), m.group(2), int(m.group(3))
    standard_id = f"{family}-{number}"
    return standard_id, family, version


def _extract_title(pages: list[dict]) -> str | None:
    """Scan the first page for a 'Title:' line; return the text or None."""
    if not pages:
        return None
    for line in pages[0]["text"].splitlines():
        m = _TITLE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def main(pdf_path: str) -> None:
    path = Path(pdf_path).resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    standard_id, standard_family, version = _parse_filename(path.stem)
    print(f"Ingesting {standard_id} v{version} from {path.name} ...")

    # -- Parse --
    pages = extract_pages(str(path))
    print(f"  Extracted {len(pages)} pages")

    # -- Chunk --
    chunks = chunk_standard(pages)
    print(f"  Produced {len(chunks)} chunks")

    # -- Classify + crossref (enrich chunks in-place) --
    for chunk in chunks:
        chunk["obligation_strength"] = classify_obligation(chunk["body"])
        chunk["related_standards"] = extract_related_standards(chunk["body"])

    # -- Embed --
    chunks = embed_chunks(chunks, standard_id)
    print(f"  Embedded {len(chunks)} chunks")

    # -- Build metadata dict --
    title = _extract_title(pages)
    meta = {
        "standard_id": standard_id,
        "version": version,
        "standard_family": standard_family,
        "title": title,
        "source_hash": file_sha256(str(path)),
        "file_path": str(path),
        "total_chunks": len(chunks),
        "effective_date": None,
        "superseded_by": None,
        "is_external": False,
    }

    # -- Upsert --
    database_url = os.environ["DATABASE_URL"]
    with psycopg.connect(database_url) as conn:
        upsert_document(conn, meta, chunks)

    print(
        f"\nDone: {standard_id} v{version} | "
        f"{len(pages)} pages | {len(chunks)} chunks | "
        f"title={title!r}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.run <pdf_path>")
        sys.exit(1)
    main(sys.argv[1])
