"""
Export LogReg out-of-fold predictions as a versioned TSV artifact.

Source data:  classifier/labels/labeled_v1.tsv
Output:       classifier/predictions/logreg_oof_v1.tsv

Pipeline is byte-identical to classifier/baselines/tfidf_logreg_baseline.py:
  - TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
  - LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - Vectorizer fit on train fold only — no leakage

Reproducibility contract: OOF F1 vs human_label must equal 0.913 (3 dp).
The script exits non-zero if this assertion fails and writes nothing.

The output TSV is a durable artifact consumed by downstream steps
(e.g. populating obligation_strength_v2) without re-running CV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

_LABELED_TSV  = Path("classifier/labels/labeled_v1.tsv")
_PREDICTIONS_DIR = Path("classifier/predictions")
_OUT_TSV      = _PREDICTIONS_DIR / "logreg_oof_v1.tsv"

_EXPECTED_F1  = 0.913


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _f1(actual: list[int], predicted: list[int]) -> float:
    tp = fp = fn = 0
    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def main() -> None:
    rows = _load(_LABELED_TSV)

    chunk_ids:   list[str] = [r["chunk_id"]    for r in rows]
    bodies:      list[str] = [r["body_preview"] for r in rows]
    regex_labels: list[str] = [r["regex_label"] for r in rows]
    human_labels: list[int] = [int(r["human_label"]) for r in rows]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds: list[int] = [0] * len(rows)

    for train_idx, val_idx in cv.split(bodies, human_labels):
        X_train = [bodies[i]       for i in train_idx]
        y_train = [human_labels[i] for i in train_idx]
        X_val   = [bodies[i]       for i in val_idx]

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="unicode",
        )
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

        X_train_vec = vec.fit_transform(X_train)
        X_val_vec   = vec.transform(X_val)
        clf.fit(X_train_vec, y_train)
        fold_preds = clf.predict(X_val_vec).tolist()

        for orig_idx, pred in zip(val_idx, fold_preds):
            oof_preds[orig_idx] = pred

    # ------------------------------------------------------------------ #
    # Reproducibility assertion
    # ------------------------------------------------------------------ #
    actual_f1 = round(_f1(human_labels, oof_preds), 3)
    if actual_f1 != _EXPECTED_F1:
        print(
            f"ERROR: reproducibility check failed.\n"
            f"  expected OOF F1 = {_EXPECTED_F1}\n"
            f"  got              = {actual_f1}\n"
            "  The output TSV was NOT written.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write artifact
    # ------------------------------------------------------------------ #
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(_OUT_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chunk_id", "oof_pred", "regex_label", "human_label"])
        for cid, pred, rl, hl in zip(chunk_ids, oof_preds, regex_labels, human_labels):
            writer.writerow([cid, pred, rl, hl])

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    n_pos_pred = sum(oof_preds)
    n_disagree_regex = sum(
        1 for pred, rl in zip(oof_preds, regex_labels)
        if pred != (1 if rl == "shall" else 0)
    )

    print(f"Total rows written : {len(rows)}")
    print(f"oof_pred positives : {n_pos_pred}")
    print(f"OOF F1 vs human    : {actual_f1:.3f}  ✓ (reproducibility contract met)")
    print(f"oof_pred ≠ regex   : {n_disagree_regex}  chunks")
    print(f"Output             : {_OUT_TSV}")


if __name__ == "__main__":
    main()
