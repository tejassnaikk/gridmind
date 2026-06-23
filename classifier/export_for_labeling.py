# NOTE: tab characters inside body text are stripped by regexp_replace in the SQL
# query (replaced with a space). This is intentional and lossy — the TSV format
# requires tabs exclusively as column separators.

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

_LABELS_DIR = Path(__file__).parent / "labels"
_OUT_FILE = _LABELS_DIR / "to_label.tsv"
_BODY_PREVIEW_LEN = 400

_QUERY = """
SELECT
    sc.id::text                                          AS chunk_id,
    sdm.standard_id                                      AS standard_id,
    sdm.version                                          AS version,
    COALESCE(sc.requirement_id, '')                      AS requirement_id,
    sc.obligation_strength                               AS regex_label,
    COALESCE(sc.page_number, 0)                          AS page,
    regexp_replace(sc.body, E'[\\n\\r\\t]+', ' ', 'g')   AS body_oneline
FROM standard_chunks sc
JOIN standard_document_metadata sdm ON sc.document_id = sdm.document_id
ORDER BY sdm.standard_id, sdm.version, sc.chunk_index;
"""

_HEADER = (
    "chunk_id\tstandard\tversion\treq_id\tregex_label\tpage\thuman_label\tbody_preview"
)


def _preview(body: str) -> str:
    if len(body) > _BODY_PREVIEW_LEN:
        return body[:_BODY_PREVIEW_LEN] + "..."
    return body


def main() -> None:
    _LABELS_DIR.mkdir(parents=True, exist_ok=True)

    database_url = os.environ["DATABASE_URL"]
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(_QUERY)
            rows = cur.fetchall()

    label_counts: Counter[str] = Counter()
    lines: list[str] = [_HEADER]

    for chunk_id, standard_id, version, req_id, regex_label, page, body_oneline in rows:
        regex_str = regex_label if regex_label is not None else ""
        label_counts[regex_str] += 1

        tsv_row = "\t".join(
            [
                chunk_id,
                standard_id,
                str(version),
                req_id,
                regex_str,
                str(page),
                "",                         # human_label — blank for labeler
                _preview(body_oneline),
            ]
        )
        lines.append(tsv_row)

    _OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = len(rows)
    size_kb = _OUT_FILE.stat().st_size / 1024

    print(f"Exported {total} chunks to {_OUT_FILE}")
    print(f"File size: {size_kb:.1f} KB")
    print()
    print("Counts by regex_label:")
    for label in ("shall", "should", "may", "informational", ""):
        count = label_counts.get(label, 0)
        display = label if label else "(null/empty)"
        print(f"  {display:<16} {count}")
    print()
    print(
        "Open classifier/labels/to_label.tsv in VS Code; "
        "fill the human_label column with 1 or 0 for each row."
    )


if __name__ == "__main__":
    main()
