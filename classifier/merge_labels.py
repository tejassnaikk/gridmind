"""
Merge human labels from human_labels_v1.txt into to_label.tsv.

Usage:
    python -m classifier.merge_labels

Writes classifier/labels/labeled_v1.tsv with human_label column filled.
Exits non-zero if any validation check fails; never writes output on failure.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_LABELS_DIR = Path("classifier/labels")
_LABELS_FILE = _LABELS_DIR / "human_labels_v1.txt"
_TSV_FILE = _LABELS_DIR / "to_label.tsv"
_OUT_FILE = _LABELS_DIR / "labeled_v1.tsv"

_EXPECTED_HEADER = [
    "chunk_id", "standard", "version", "req_id",
    "regex_label", "page", "human_label", "body_preview",
]

_VALID_LABELS = {"0", "1"}
_REGEX_OBLIGATION_LABELS = {"shall", "should", "may", "informational"}


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_labels() -> dict[str, int]:
    if not _LABELS_FILE.exists():
        _fail(f"{_LABELS_FILE} not found")

    labels: dict[str, int] = {}
    for lineno, raw in enumerate(_LABELS_FILE.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            _fail(f"{_LABELS_FILE}:{lineno}: expected '<prefix> <0|1>', got {raw!r}")
        prefix, label_str = parts
        if label_str not in _VALID_LABELS:
            _fail(f"{_LABELS_FILE}:{lineno}: label must be 0 or 1, got {label_str!r}")
        if prefix in labels:
            _fail(f"{_LABELS_FILE}:{lineno}: duplicate prefix {prefix!r}")
        labels[prefix] = int(label_str)
    return labels


def _load_tsv() -> tuple[list[str], list[dict[str, str]]]:
    if not _TSV_FILE.exists():
        _fail(f"{_TSV_FILE} not found")

    with open(_TSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        if list(header) != _EXPECTED_HEADER:
            _fail(
                f"Unexpected TSV header.\n"
                f"  expected: {_EXPECTED_HEADER}\n"
                f"  got:      {list(header)}"
            )
        rows = list(reader)
    return list(header), rows


def _build_tsv_index(rows: list[dict[str, str]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, row in enumerate(rows):
        prefix = row["chunk_id"][:8]
        if prefix in index:
            _fail(
                f"Two TSV rows share the same 8-char prefix {prefix!r}: "
                f"rows {index[prefix]} and {i}"
            )
        index[prefix] = i
    return index


def _validate(
    labels: dict[str, int],
    tsv_index: dict[str, int],
) -> None:
    errors: list[str] = []

    orphans = sorted(set(labels) - set(tsv_index))
    if orphans:
        errors.append(
            "Label prefixes with no matching TSV row:\n"
            + "".join(f"  {p}\n" for p in orphans)
        )

    unlabeled = sorted(set(tsv_index) - set(labels))
    if unlabeled:
        errors.append(
            "TSV rows with no label entry:\n"
            + "".join(f"  {p}\n" for p in unlabeled)
        )

    if len(labels) != len(tsv_index):
        errors.append(
            f"Count mismatch: {len(labels)} labels vs {len(tsv_index)} TSV rows"
        )

    if errors:
        _fail("\n".join(errors))


def _print_summary(rows: list[dict[str, str]]) -> None:
    total = len(rows)
    class_counts: Counter[int] = Counter(int(r["human_label"]) for r in rows)

    print(f"Total chunks labeled: {total}")
    print(f"  human=0: {class_counts[0]}    human=1: {class_counts[1]}")
    print()

    # 2×4 disagreement grid: human_label × regex_label
    obligation_cols = ["shall", "should", "may", "informational"]
    grid: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    for row in rows:
        hl = int(row["human_label"])
        rl = row["regex_label"] if row["regex_label"] in _REGEX_OBLIGATION_LABELS else "informational"
        grid[hl][rl] += 1

    col_w = 14
    header_line = f"{'':12}" + "".join(f"{c:>{col_w}}" for c in obligation_cols)
    print(header_line)
    print("-" * len(header_line))
    for hl in (0, 1):
        label_str = f"human={hl}"
        row_str = f"{label_str:<12}" + "".join(
            f"{grid[hl][c]:>{col_w}}" for c in obligation_cols
        )
        print(row_str)
    print()

    # Disagreement list: binary collapse of regex (shall -> 1, else -> 0)
    def regex_binary(rl: str) -> int:
        return 1 if rl == "shall" else 0

    under_tags: list[dict[str, str]] = []  # human=1, regex=0
    over_tags: list[dict[str, str]] = []   # human=0, regex=1

    for row in rows:
        hl = int(row["human_label"])
        rb = regex_binary(row["regex_label"])
        if hl == 1 and rb == 0:
            under_tags.append(row)
        elif hl == 0 and rb == 1:
            over_tags.append(row)

    def _show_group(title: str, group: list[dict[str, str]]) -> None:
        print(f"{title} ({len(group)}):")
        if not group:
            print("  (none)")
        for r in group:
            req = r["req_id"] or "—"
            preview = r["body_preview"][:80]
            print(f"  {r['chunk_id']}  {r['standard']} v{r['version']} {req}")
            print(f"    {preview}")
        print()

    _show_group("human=1, regex=0  [regex under-tags]", under_tags)
    _show_group("human=0, regex=1  [regex over-tags]", over_tags)


def main() -> None:
    labels = _load_labels()
    header, rows = _load_tsv()
    tsv_index = _build_tsv_index(rows)
    _validate(labels, tsv_index)

    # Apply labels in original TSV order
    for row in rows:
        prefix = row["chunk_id"][:8]
        row["human_label"] = str(labels[prefix])

    # Write output only after all validation passes
    with open(_OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    _print_summary(rows)
    print(f"Output: {_OUT_FILE}")


if __name__ == "__main__":
    main()
