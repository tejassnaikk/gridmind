"""
TF-IDF + Logistic Regression baseline for the GridMind obligation classifier.

Feature: body_preview column (first 400 chars of chunk body).
Target:  human_label column (0/1).
Evaluation: 5-fold stratified cross-validation, out-of-fold predictions.

Usage:
    python -m classifier.baselines.tfidf_logreg_baseline
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
except ImportError:
    print(
        "ERROR: scikit-learn is not installed.\n"
        "Install it with:\n"
        "    pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)

_LABELED_TSV = Path("classifier/labels/labeled_v1.tsv")
_REGEX_F1 = 0.727  # measured by classifier/baselines/regex_baseline.py

_EXPECTED_TOTAL = 96
_EXPECTED_POS   = 23
_EXPECTED_NEG   = 73


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _metrics(actual: list[int], predicted: list[int]) -> tuple[int, int, int, int, float, float, float, float]:
    tn = fp = fn = tp = 0
    for a, p in zip(actual, predicted):
        if a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1
        else:
            tp += 1
    total     = tn + fp + fn + tp
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return tn, fp, fn, tp, accuracy, precision, recall, f1


def main() -> None:
    rows = _load(_LABELED_TSV)

    bodies: list[str] = []
    labels: list[int] = []
    for row in rows:
        bodies.append(row["body_preview"])
        labels.append(int(row["human_label"]))

    total    = len(labels)
    n_pos    = sum(labels)
    n_neg    = total - n_pos

    if total != _EXPECTED_TOTAL or n_pos != _EXPECTED_POS or n_neg != _EXPECTED_NEG:
        print(
            f"ERROR: unexpected dataset shape.\n"
            f"  expected {_EXPECTED_TOTAL} rows / {_EXPECTED_POS} pos / {_EXPECTED_NEG} neg\n"
            f"  got      {total} rows / {n_pos} pos / {n_neg} neg",
            file=sys.stderr,
        )
        sys.exit(1)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Per-fold F1 (compute manually so we can show variance)
    fold_f1s: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(bodies, labels)):
        X_train = [bodies[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val   = [bodies[i] for i in val_idx]
        y_val   = [labels[i] for i in val_idx]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val).tolist()

        _, _, fn_f, tp_f, _, prec_f, rec_f, f1_f = _metrics(y_val, preds)
        fold_f1s.append(f1_f)

    # Out-of-fold predictions (full aggregate)
    oof_preds: list[int] = cross_val_predict(pipe, bodies, labels, cv=cv).tolist()

    tn, fp, fn, tp, accuracy, precision, recall, f1 = _metrics(labels, oof_preds)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("=" * 56)
    print("  GridMind TF-IDF + LogReg — Obligation Classifier")
    print("=" * 56)
    print(f"\nTotal chunks evaluated: {total}  ({n_pos} pos / {n_neg} neg)")
    print("Evaluation: 5-fold stratified cross-validation (out-of-fold)")

    print("\n── Confusion Matrix ──────────────────────────────────")
    print(f"                  Predicted 0    Predicted 1    Total")
    print(f"  Actual 0  (TN)  {tn:>10}     {fp:>10}  {tn+fp:>7}")
    print(f"  Actual 1  (FN)  {fn:>10}     {tp:>10}  {fn+tp:>7}")
    print(f"  Total           {tn+fn:>10}     {fp+tp:>10}  {total:>7}")

    print("\n── Metrics (aggregate out-of-fold) ───────────────────")
    print(f"  Accuracy   {accuracy:.3f}")
    print(f"  Precision  {precision:.3f}   (TP / predicted-positive)")
    print(f"  Recall     {recall:.3f}   (TP / actual-positive)")
    print(f"  F1         {f1:.3f}")

    print("\n── Per-fold F1 (variance check) ──────────────────────")
    fold_str = "  " + "  ".join(f"fold{i+1}={v:.3f}" for i, v in enumerate(fold_f1s))
    mean_f1  = sum(fold_f1s) / len(fold_f1s)
    print(fold_str)
    print(f"  mean={mean_f1:.3f}   (unweighted; aggregate OOF F1 above is authoritative)")

    # ------------------------------------------------------------------ #
    # Disagreements against human labels
    # ------------------------------------------------------------------ #
    false_positives: list[dict[str, str]] = []  # LogReg=1, human=0
    false_negatives: list[dict[str, str]] = []  # LogReg=0, human=1

    for row, pred in zip(rows, oof_preds):
        actual = int(row["human_label"])
        if pred == 1 and actual == 0:
            false_positives.append(row)
        elif pred == 0 and actual == 1:
            false_negatives.append(row)

    def _show_group(title: str, group: list[dict[str, str]]) -> None:
        print(f"\n{title} ({len(group)}):")
        if not group:
            print("  (none)")
            return
        for r in group:
            req = r["req_id"] or "—"
            preview = r["body_preview"][:80]
            print(f"  {r['chunk_id']}  {r['standard']} v{r['version']} {req}")
            print(f"    {preview}")

    print("\n── Disagreements vs human labels ─────────────────────")
    _show_group("LogReg=1, human=0  [false positives]", false_positives)
    _show_group("LogReg=0, human=1  [false negatives]", false_negatives)

    # ------------------------------------------------------------------ #
    # Comparison table
    # ------------------------------------------------------------------ #
    print("\n── Comparison vs regex baseline ──────────────────────")
    print(f"  {'Model':<32}  {'F1':>6}")
    print(f"  {'-'*32}  {'------'}")
    print(f"  {'Regex (shall → 1)':<32}  {_REGEX_F1:>6.3f}")
    print(f"  {'TF-IDF + LogReg (OOF CV)':<32}  {f1:>6.3f}")
    delta = f1 - _REGEX_F1
    direction = "▲" if delta > 0 else "▼"
    print(f"  {'Delta':<32}  {direction} {abs(delta):.3f}")
    print()


if __name__ == "__main__":
    main()
