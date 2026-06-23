"""
Regex baseline for the GridMind obligation classifier.

Prediction rule: 1 if regex_label == 'shall', else 0.
Ground truth:    human_label column in labeled_v1.tsv.

Usage:
    python -m classifier.baselines.regex_baseline
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_LABELED_TSV = Path("classifier/labels/labeled_v1.tsv")


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _predict(regex_label: str) -> int:
    return 1 if regex_label == "shall" else 0


def main() -> None:
    rows = _load(_LABELED_TSV)
    total = len(rows)

    tn = fp = fn = tp = 0
    for row in rows:
        actual = int(row["human_label"])
        predicted = _predict(row["regex_label"])
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("=" * 52)
    print("  GridMind Regex Baseline — Obligation Classifier")
    print("=" * 52)
    print(f"\nTotal chunks evaluated: {total}")

    print("\n── Confusion Matrix ──────────────────────────────")
    print(f"                  Predicted 0    Predicted 1    Total")
    print(f"  Actual 0  (TN)  {tn:>10}     {fp:>10}  {tn+fp:>7}")
    print(f"  Actual 1  (FN)  {fn:>10}     {tp:>10}  {fn+tp:>7}")
    print(f"  Total           {tn+fn:>10}     {fp+tp:>10}  {total:>7}")

    print("\n── Metrics ───────────────────────────────────────")
    print(f"  Accuracy   {accuracy:.3f}")
    print(f"  Precision  {precision:.3f}   (TP / predicted-positive)")
    print(f"  Recall     {recall:.3f}   (TP / actual-positive)")
    print(f"  F1         {f1:.3f}")

    print("\n── Disagreement summary ──────────────────────────")
    print(f"  Regex under-tags (human=1, regex=0): {fn}  chunks")
    print(f"  Regex over-tags  (human=0, regex=1): {fp}  chunks")
    print(f"  Total disagreements: {fn + fp}  ({(fn+fp)/total*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
