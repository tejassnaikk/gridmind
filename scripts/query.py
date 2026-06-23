"""
CLI wrapper for GridMind retrieval.

Usage:
    python -m scripts.query "What must a Responsible Entity do to identify
                              cyber security risks in the supply chain?"

    python -m scripts.query "vendor remote access controls" --k 10
"""

from __future__ import annotations

import argparse
import sys

from retrieval.crossref_expand import query_with_expansion
from retrieval.query import query

BODY_PREVIEW = 200


def _truncate(text: str) -> str:
    if len(text) <= BODY_PREVIEW:
        return text
    return text[:BODY_PREVIEW].rstrip() + " …"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the GridMind retrieval pipeline."
    )
    parser.add_argument("question", help="Natural-language question to retrieve for")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="N",
        help="Number of results to return (default: config final_k=5)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Enable cross-reference expansion (query_with_expansion strategy)",
    )
    args = parser.parse_args()

    if args.expand:
        results = query_with_expansion(args.question, k=args.k)
    else:
        results = query(args.question, k=args.k)

    if not results:
        print("No results found.")
        sys.exit(0)

    print()
    for r in results:
        req = r["requirement_id"] or "—"
        page = r["page_number"] or "?"
        xref_tag = "  [xref]" if r.get("from_crossref") else ""
        print(
            f"[{r['rank']}] {r['standard_id']} v{r['version']}  "
            f"{req}  (p.{page})  score={r['score']:.4f}{xref_tag}"
        )
        print(f"    {_truncate(r['body'])}")
        print()


if __name__ == "__main__":
    main()
