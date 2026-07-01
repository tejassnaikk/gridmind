"""
DistilBERT fine-tune baseline for the GridMind obligation classifier.

Feature: body_preview column (first 400 chars of chunk body).
Target:  human_label column (0/1).
Evaluation: 5-fold stratified cross-validation, out-of-fold predictions.
Protocol matches tfidf_logreg_baseline.py for direct comparison.

Usage:
    python -m classifier.baselines.distilbert_baseline
"""

from __future__ import annotations

import csv
import contextlib
import io
import warnings
warnings.filterwarnings("ignore")

import transformers  # noqa: E402
transformers.logging.set_verbosity_error()
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

_LABELED_TSV = Path("classifier/labels/labeled_v1.tsv")
_REGEX_F1    = 0.727
_LOGREG_F1   = 0.913

_EXPECTED_TOTAL = 96
_EXPECTED_POS   = 23
_EXPECTED_NEG   = 73

_MODEL_NAME = "distilbert-base-uncased"


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _metrics(
    actual: list[int], predicted: list[int]
) -> tuple[int, int, int, int, float, float, float, float]:
    tn = fp = fn = tp = 0
    for a, p in zip(actual, predicted):
        if   a == 0 and p == 0: tn += 1
        elif a == 0 and p == 1: fp += 1
        elif a == 1 and p == 0: fn += 1
        else:                   tp += 1
    total     = tn + fp + fn + tp
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return tn, fp, fn, tp, accuracy, precision, recall, f1


class _ObligationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    if torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
        print(
            "WARNING: MPS not available. Training on CPU — expect ~30+ min runtime.",
            file=sys.stderr,
        )
    print(f"Device: {device_name}", file=sys.stderr)

    rows = _load(_LABELED_TSV)
    bodies: list[str] = [r["body_preview"] for r in rows]
    labels: list[int] = [int(r["human_label"]) for r in rows]

    total = len(labels)
    n_pos = sum(labels)
    n_neg = total - n_pos

    if total != _EXPECTED_TOTAL or n_pos != _EXPECTED_POS or n_neg != _EXPECTED_NEG:
        print(
            f"ERROR: unexpected dataset shape.\n"
            f"  expected {_EXPECTED_TOTAL} rows / {_EXPECTED_POS} pos / {_EXPECTED_NEG} neg\n"
            f"  got      {total} rows / {n_pos} pos / {n_neg} neg",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = DistilBertTokenizerFast.from_pretrained(_MODEL_NAME)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds: list[int] = [0] * total
    fold_f1s:  list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(bodies, labels)):
        print(f"[fold {fold_idx+1}/5] training...", file=sys.stderr, flush=True)

        X_train = [bodies[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val   = [bodies[i] for i in val_idx]
        y_val   = [labels[i] for i in val_idx]

        enc_train = tokenizer(
            X_train, max_length=256, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_val = tokenizer(
            X_val, max_length=256, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        train_ds = _ObligationDataset(enc_train, y_train)
        val_ds   = _ObligationDataset(enc_val,   y_val)

        model = DistilBertForSequenceClassification.from_pretrained(
            _MODEL_NAME, num_labels=2
        )

        args = TrainingArguments(
            output_dir=f"/tmp/distilbert_fold_{fold_idx}",
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()

        raw = trainer.predict(val_ds)
        fold_preds = np.argmax(raw.predictions, axis=-1).tolist()

        for orig_idx, pred in zip(val_idx, fold_preds):
            oof_preds[orig_idx] = pred

        _, _, _, _, _, _, _, f1_f = _metrics(y_val, fold_preds)
        fold_f1s.append(f1_f)
        print(f"[fold {fold_idx+1}/5] F1={f1_f:.3f}", file=sys.stderr, flush=True)

    tn, fp, fn, tp, accuracy, precision, recall, f1 = _metrics(labels, oof_preds)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("=" * 56)
    print("  GridMind DistilBERT Fine-tune — Obligation Classifier")
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
    # Disagreements
    # ------------------------------------------------------------------ #
    false_positives: list[dict[str, str]] = []
    false_negatives: list[dict[str, str]] = []

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
            req     = r["req_id"] or "—"
            preview = r["body_preview"][:80]
            print(f"  {r['chunk_id']}  {r['standard']} v{r['version']} {req}")
            print(f"    {preview}")

    print("\n── Disagreements vs human labels ─────────────────────")
    _show_group("DistilBERT=1, human=0  [false positives]", false_positives)
    _show_group("DistilBERT=0, human=1  [false negatives]", false_negatives)

    # ------------------------------------------------------------------ #
    # Comparison table
    # ------------------------------------------------------------------ #
    delta_regex  = f1 - _REGEX_F1
    delta_logreg = f1 - _LOGREG_F1

    print("\n── Comparison: all three baselines ───────────────────")
    print(f"  {'Model':<40}  {'F1':>6}")
    print(f"  {'-'*40}  {'------'}")
    print(f"  {'Regex (shall → 1)':<40}  {_REGEX_F1:>6.3f}")
    print(f"  {'TF-IDF + LogReg (OOF CV)':<40}  {_LOGREG_F1:>6.3f}")
    print(f"  {'DistilBERT fine-tune (OOF CV)':<40}  {f1:>6.3f}")
    print(f"  {'Delta vs regex baseline':<40}  {'▲' if delta_regex  > 0 else '▼'} {abs(delta_regex):.3f}")
    print(f"  {'Delta vs LogReg baseline':<40}  {'▲' if delta_logreg > 0 else '▼'} {abs(delta_logreg):.3f}")
    print()


if __name__ == "__main__":
    main()
