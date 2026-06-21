"""
Usage:  python ingestion/inspect_chunks.py path/to/standard.pdf

Runs extract_pages, chunk_standard, classify_obligation, and
extract_related_standards, then prints every chunk for quality review.
No embedding or DB writes.
"""

import sys

from ingestion.chunk import chunk_standard
from ingestion.classify import classify_obligation
from ingestion.crossref import extract_related_standards
from ingestion.parse import extract_pages

PREVIEW_CHARS = 120
MAX_REFS_SHOWN = 4  # truncate long cross-ref lists in the display


def _label(chunk: dict, seen_requirement: bool) -> str:
    if chunk["requirement_id"] is not None:
        return chunk["requirement_id"]
    return "preamble" if not seen_requirement else "section"


def _fmt_refs(refs: list[str]) -> str:
    if not refs:
        return "—"
    if len(refs) <= MAX_REFS_SHOWN:
        return ",".join(refs)
    return ",".join(refs[:MAX_REFS_SHOWN]) + f" +{len(refs) - MAX_REFS_SHOWN}"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ingestion/inspect_chunks.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    pages = extract_pages(pdf_path)
    chunks = chunk_standard(pages)

    print(f"Pages : {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print()
    print(
        f"{'idx':>4}  {'req_id':<14}  {'pg':>3}  {'ch':>5}  "
        f"{'oblig':<14}  {'xrefs':<28}  preview"
    )
    print("-" * 110)

    seen_req = False
    for chunk in chunks:
        if chunk["requirement_id"] is not None:
            seen_req = True
        label = _label(chunk, seen_req)
        oblig = classify_obligation(chunk["body"])
        refs = extract_related_standards(chunk["body"])
        preview = chunk["body"][:PREVIEW_CHARS].replace("\n", " ")
        print(
            f"{chunk['chunk_index']:>4}  {label:<14}  "
            f"{chunk['page_number']:>3}  {len(chunk['body']):>5}  "
            f"{oblig:<14}  {_fmt_refs(refs):<28}  {preview!r}"
        )


if __name__ == "__main__":
    main()
