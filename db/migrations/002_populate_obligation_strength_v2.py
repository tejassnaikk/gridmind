"""
Populate the obligation_strength_v2 column added by
db/migrations/002_add_obligation_strength_v2.sql.

Input artifact:  classifier/predictions/logreg_oof_v1.tsv
                 Columns: chunk_id, oof_pred, regex_label, human_label

Output effect:   UPDATE standard_chunks SET obligation_strength_v2 = <value>
                 for every chunk_id present in the TSV (96 rows).

Mapping (locked for the experiment — do not vary):

    regex_label     oof_pred    obligation_strength_v2
    -------------------------------------------------------
    shall           1           shall
    should          1           should
    may             1           may
    informational   1           should
    any             0           informational

Rationale: when the classifier predicts 1 (obligation), preserve the
lexical register captured by the regex (shall/should/may); promote
informational to 'should' because the classifier judged it obligatory
but the regex found no strong modal. When oof_pred=0, collapse to
'informational' regardless of regex_label — the classifier overrides.

Idempotent: all operations are UPDATEs; re-running produces the same state.

Usage:
    python -m db.migrations.002_populate_obligation_strength_v2
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import psycopg

_TSV = Path("classifier/predictions/logreg_oof_v1.tsv")

_VALID_REGEX_LABELS = {"shall", "should", "may", "informational"}

# Mapping: (regex_label, oof_pred) -> obligation_strength_v2
# oof_pred=0 always maps to 'informational'; handled inline.
_PRED1_MAP: dict[str, str] = {
    "shall":         "shall",
    "should":        "should",
    "may":           "may",
    "informational": "should",
}


def _resolve(chunk_id: str, regex_label: str, oof_pred: int) -> str:
    if oof_pred == 0:
        return "informational"
    if regex_label not in _VALID_REGEX_LABELS:
        raise ValueError(
            f"chunk_id {chunk_id!r}: unexpected regex_label {regex_label!r} "
            f"with oof_pred=1; expected one of {sorted(_VALID_REGEX_LABELS)}"
        )
    return _PRED1_MAP[regex_label]


def _load_tsv(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (chunk_id, oof_pred, regex_label)."""
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    rows: list[tuple[str, int, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append((row["chunk_id"], int(row["oof_pred"]), row["regex_label"]))
    return rows


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise KeyError(
            "DATABASE_URL environment variable is not set. "
            "Export it before running this script, e.g.:\n"
            "  export DATABASE_URL=postgresql://gridmind:gridmind@localhost:5432/gridmind_embeddings"
        )

    tsv_rows = _load_tsv(_TSV)

    # Resolve v2 label for each row; fail fast on data errors
    updates: list[tuple[str, str]] = []  # (obligation_strength_v2, chunk_id)
    for chunk_id, oof_pred, regex_label in tsv_rows:
        v2 = _resolve(chunk_id, regex_label, oof_pred)
        updates.append((v2, chunk_id))

    # ------------------------------------------------------------------ #
    # Single-transaction UPDATE
    # ------------------------------------------------------------------ #
    missing: list[str] = []

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for v2_label, chunk_id in updates:
                cur.execute(
                    "UPDATE standard_chunks SET obligation_strength_v2 = %s WHERE id = %s::uuid",
                    (v2_label, chunk_id),
                )
                if cur.rowcount == 0:
                    missing.append(chunk_id)

        if missing:
            conn.rollback()
            raise RuntimeError(
                f"Aborting: {len(missing)} chunk_id(s) from TSV not found in DB "
                f"(transaction rolled back):\n"
                + "\n".join(f"  {cid}" for cid in missing)
            )

        conn.commit()

    rows_updated = len(updates)
    print(f"Rows updated: {rows_updated}")

    # ------------------------------------------------------------------ #
    # Verification queries
    # ------------------------------------------------------------------ #
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM standard_chunks WHERE obligation_strength_v2 IS NOT NULL"
            )
            total_non_null = cur.fetchone()[0]
            print(f"Non-null obligation_strength_v2 rows: {total_non_null}  (expect 96)")

            cur.execute(
                """
                SELECT obligation_strength_v2, COUNT(*) AS n
                FROM standard_chunks
                WHERE obligation_strength_v2 IS NOT NULL
                GROUP BY obligation_strength_v2
                ORDER BY n DESC
                """
            )
            print("\nDistribution of obligation_strength_v2:")
            for label, count in cur.fetchall():
                print(f"  {label:<16} {count}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM standard_chunks
                WHERE obligation_strength_v2 IS NOT NULL
                  AND obligation_strength != obligation_strength_v2
                """
            )
            n_disagree = cur.fetchone()[0]
            print(f"\nChunks where obligation_strength != obligation_strength_v2: {n_disagree}  (expect 12)")


if __name__ == "__main__":
    main()
