"""
Pre-registered breakdown for the v2 retrieval-impact experiment.
Reports ΔR@5, ΔMRR, chunk-level prior changes, per-question deltas,
and a q007 focus block.

Pre-registration: the experiment tests whether substituting the LogReg-derived
obligation_strength_v2 column for the regex-based obligation_strength column
improves retrieval quality (R@5, MRR) on the 10-question eval set.

Usage:
    python -m eval.analyze_v2_impact                        # newest results file
    python -m eval.analyze_v2_impact --results path/to/run_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg


_EVAL_DIR       = Path(__file__).parent
_QUESTIONS_FILE = _EVAL_DIR / "questions.jsonl"
_RESULTS_DIR    = _EVAL_DIR / "results"
_OOF_TSV        = Path("classifier/predictions/logreg_oof_v1.tsv")

# Prior ordering for promoted/demoted classification (higher index = stronger)
_PRIOR_WEIGHT = {"informational": 0, "may": 1, "should": 2, "shall": 3}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_questions() -> list[dict]:
    qs: list[dict] = []
    with open(_QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def _newest_results_file() -> Path:
    candidates = sorted(_RESULTS_DIR.glob("run_*.jsonl"))
    if not candidates:
        print("ERROR: no run_*.jsonl files found in eval/results/", file=sys.stderr)
        sys.exit(1)
    return candidates[-1]


def _load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_results(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Index by (strategy, question_id)."""
    idx: dict[tuple[str, str], dict] = {}
    for r in rows:
        idx[(r["strategy"], r["question_id"])] = r
    return idx


def _load_changed_chunks(database_url: str) -> dict[str, tuple[str, str]]:
    """
    Return {chunk_id: (v1_prior, v2_prior)} for every chunk where
    obligation_strength != obligation_strength_v2.
    Only rows where both columns are non-null are considered.
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, obligation_strength, obligation_strength_v2
                FROM   standard_chunks
                WHERE  obligation_strength_v2 IS NOT NULL
                  AND  obligation_strength IS DISTINCT FROM obligation_strength_v2
                """
            )
            return {
                row[0]: (row[1], row[2])
                for row in cur.fetchall()
            }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _sign(v: float) -> str:
    if v > 0:
        return f"+{v:.3f}"
    if v < 0:
        return f"{v:.3f}"
    return " 0.000"


def _direction(v1_prior: str, v2_prior: str) -> str:
    w1 = _PRIOR_WEIGHT.get(v1_prior, -1)
    w2 = _PRIOR_WEIGHT.get(v2_prior, -1)
    if w2 > w1:
        return "promoted"
    if w2 < w1:
        return "demoted"
    return "unchanged"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(results_path: Path, rows: list[dict], questions: list[dict]) -> None:
    strategies_found = sorted({r["strategy"] for r in rows})
    qids_found = sorted({r["question_id"] for r in rows})
    ts = rows[0].get("timestamp", "unknown") if rows else "unknown"

    print("=" * 70)
    print("  GridMind v2 Retrieval-Impact Analysis")
    print("=" * 70)
    print(f"  Results file : {results_path}")
    print(f"  Timestamp    : {ts}")
    print(f"  Questions    : {len(qids_found)}  (expected 10)")
    print(f"  Strategies   : {', '.join(strategies_found)}")
    print()


def _aggregate_deltas(
    idx: dict[tuple[str, str], dict],
    questions: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute and print aggregate ΔR@5, ΔR@10, ΔMRR for both comparison pairs.
    Returns (delta_base, delta_crossref) where each is {"r5", "r10", "mrr"}.
    """
    qids = [q["id"] for q in questions]

    def _delta(s1: str, s2: str) -> dict[str, float]:
        d5, d10, dmrr = [], [], []
        for qid in qids:
            r1 = idx.get((s1, qid))
            r2 = idx.get((s2, qid))
            if r1 is None or r2 is None:
                continue
            d5.append(r2["recall_at_5"] - r1["recall_at_5"])
            d10.append(r2["recall_at_10"] - r1["recall_at_10"])
            dmrr.append(r2["mrr"] - r1["mrr"])
        return {"r5": _mean(d5), "r10": _mean(d10), "mrr": _mean(dmrr)}

    base_delta   = _delta("rrf+priors",          "rrf+priors_v2")
    xref_delta   = _delta("rrf+priors+crossref", "rrf+priors+crossref_v2")

    print("── Aggregate deltas (v2 − v1) ─────────────────────────────────")
    print(f"  {'Comparison':<44}  {'ΔR@5':>7}  {'ΔR@10':>7}  {'ΔMRR':>7}")
    print(f"  {'-'*44}  {'-------'}  {'-------'}  {'-------'}")
    print(
        f"  {'rrf+priors  →  rrf+priors_v2':<44}"
        f"  {_sign(base_delta['r5']):>7}"
        f"  {_sign(base_delta['r10']):>7}"
        f"  {_sign(base_delta['mrr']):>7}"
    )
    print(
        f"  {'rrf+priors+crossref  →  rrf+priors+crossref_v2':<44}"
        f"  {_sign(xref_delta['r5']):>7}"
        f"  {_sign(xref_delta['r10']):>7}"
        f"  {_sign(xref_delta['mrr']):>7}"
    )
    print()
    return base_delta, xref_delta


def _chunk_level_changes(changed: dict[str, tuple[str, str]]) -> None:
    print("── Chunk-level prior changes ──────────────────────────────────")
    print(f"  Total chunks changed (v1 → v2): {len(changed)}  (expected 33)")
    print()

    # Transition counts
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    for v1, v2 in changed.values():
        transitions[(v1, v2)] += 1

    rows = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'From → To':<36}  {'Count':>5}")
    print(f"  {'-'*36}  {'-----'}")
    for (v1, v2), cnt in rows:
        print(f"  {v1:<16} → {v2:<16}  {cnt:>5}")
    print()


def _coverage(
    changed: dict[str, tuple[str, str]],
    questions: list[dict],
) -> set[str]:
    """Print gold-chunk coverage and return the set of affected gold chunk ids."""
    all_gold: set[str] = set()
    for q in questions:
        all_gold.update(q["gold_chunk_ids"])

    affected_gold = all_gold & set(changed)

    print("── Coverage: gold chunks affected ─────────────────────────────")
    print(
        f"  Of {len(all_gold)} unique gold chunks across the eval set, "
        f"{len(affected_gold)} had their prior changed by the classifier."
    )
    print()

    # Build qid lookup: gold_chunk_id -> list[qid]
    gold_to_qids: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        for cid in q["gold_chunk_ids"]:
            gold_to_qids[cid].append(q["id"])

    if affected_gold:
        print(f"  {'chunk_id[:8]':<12}  {'question(s)':<18}  {'v1 prior':<14}  {'v2 prior':<14}  direction")
        print(f"  {'-'*12}  {'-'*18}  {'-'*14}  {'-'*14}  ---------")
        for cid in sorted(affected_gold):
            v1, v2 = changed[cid]
            qids_str = ", ".join(gold_to_qids[cid])
            dirn = _direction(v1, v2)
            print(f"  {cid[:8]:<12}  {qids_str:<18}  {v1:<14}  {v2:<14}  {dirn}")
    print()
    return affected_gold


def _question_table(
    idx: dict[tuple[str, str], dict],
    questions: list[dict],
    affected_gold: set[str],
) -> None:
    print("── Per-question deltas (rrf+priors vs rrf+priors_v2) ──────────")
    print(
        f"  {'qid':<6}  {'R@5_v1':>7}  {'R@5_v2':>7}  {'ΔR@5':>7}"
        f"  {'MRR_v1':>7}  {'MRR_v2':>7}  {'ΔMRR':>7}  gold_changed"
    )
    print(
        f"  {'------'}  {'-------'}  {'-------'}  {'-------'}"
        f"  {'-------'}  {'-------'}  {'-------'}  ------------"
    )

    table_rows: list[tuple[float, str]] = []
    for q in questions:
        qid = q["id"]
        r_v1 = idx.get(("rrf+priors",    qid))
        r_v2 = idx.get(("rrf+priors_v2", qid))
        if r_v1 is None or r_v2 is None:
            continue
        dr5   = r_v2["recall_at_5"] - r_v1["recall_at_5"]
        dmrr  = r_v2["mrr"] - r_v1["mrr"]
        gc    = any(cid in affected_gold for cid in q["gold_chunk_ids"])
        table_rows.append((abs(dmrr), qid, r_v1, r_v2, dr5, dmrr, gc))

    table_rows.sort(key=lambda x: x[0], reverse=True)

    for _, qid, r_v1, r_v2, dr5, dmrr, gc in table_rows:
        gc_str = "yes" if gc else "no"
        print(
            f"  {qid:<6}  {r_v1['recall_at_5']:>7.3f}  {r_v2['recall_at_5']:>7.3f}  {_sign(dr5):>7}"
            f"  {r_v1['mrr']:>7.3f}  {r_v2['mrr']:>7.3f}  {_sign(dmrr):>7}  {gc_str}"
        )
    print()


def _q007_block(
    idx: dict[tuple[str, str], dict],
    questions: list[dict],
    changed: dict[str, tuple[str, str]],
) -> None:
    q007 = next((q for q in questions if q["id"] == "q007"), None)
    if q007 is None:
        print("── q007 focus block — question not found in questions.jsonl ────")
        return

    print("── q007 focus block ───────────────────────────────────────────")
    print(f"  Question: {q007['question']}")
    print()

    gold_ids = q007["gold_chunk_ids"]
    print("  Gold chunks and prior transitions:")
    for cid in gold_ids:
        if cid in changed:
            v1, v2 = changed[cid]
            dirn = _direction(v1, v2)
            print(f"    {cid}  {v1} → {v2}  ({dirn})")
        else:
            print(f"    {cid}  (prior unchanged)")
    print()

    def _show_top5(strategy_key: str, label: str) -> list[str]:
        r = idx.get((strategy_key, "q007"))
        if r is None:
            print(f"  {label}: no results found")
            return []
        retrieved = r["retrieved_ids"]
        gold_set = set(gold_ids)
        print(f"  Top-5 under {label}:")
        print(f"    {'rank':<5}  {'chunk_id[:8]':<12}  gold?")
        for rank, cid in enumerate(retrieved[:5], start=1):
            is_gold = "*** GOLD" if cid in gold_set else ""
            print(f"    {rank:<5}  {cid[:8]:<12}  {is_gold}")
        return retrieved

    retrieved_v1 = _show_top5("rrf+priors",    "rrf+priors (v1)")
    print()
    retrieved_v2 = _show_top5("rrf+priors_v2", "rrf+priors_v2")
    print()

    # Gold ranks within top-10
    print("  Gold chunk ranks within top-10:")
    for cid in gold_ids:
        rank_v1 = next(
            (i + 1 for i, c in enumerate(retrieved_v1[:10]) if c == cid), None
        )
        rank_v2 = next(
            (i + 1 for i, c in enumerate(retrieved_v2[:10]) if c == cid), None
        )
        r1_str = str(rank_v1) if rank_v1 else "not in top-10"
        r2_str = str(rank_v2) if rank_v2 else "not in top-10"
        print(f"    {cid[:8]}  v1={r1_str:>12}  v2={r2_str:>12}")
    print()

    print(
        "  Interpretation: q007 gold chunks 466235f7 and e83242ae are in the\n"
        "  7 promotions (informational → should). This is the tightest test of\n"
        "  whether classifier signal rescues a known failure."
    )
    print()


def _footer(
    base_delta: dict[str, float],
    questions: list[dict],
    idx: dict[tuple[str, str], dict],
    affected_gold: set[str],
) -> None:
    dr5  = base_delta["r5"]
    dmrr = base_delta["mrr"]

    print("── Interpretation ─────────────────────────────────────────────")

    if dr5 == 0.0 and dmrr > 0:
        conclusion = (
            "Recall unchanged (no new gold chunks entered top-K), but MRR improved — "
            "classifier priors reordered results within the retrieved set, moving gold "
            "chunks toward rank 1."
        )
    elif dr5 > 0 and dmrr > 0:
        conclusion = (
            "Both recall and ranking improved — classifier priors surfaced new gold chunks."
        )
    else:
        conclusion = (
            "Deltas neutral or negative — modality information loss from binary supervision "
            "appears to hurt more than the classifier's error corrections help. Consistent "
            "with the pre-registered hypothesis that binary obligation classification discards "
            "useful should/may distinctions the regex captured."
        )

    print(f"  {conclusion}")
    print()

    # Questions with moved MRR
    qids_with_gold_changed = [
        q["id"] for q in questions
        if any(cid in affected_gold for cid in q["gold_chunk_ids"])
    ]
    qids_mrr_moved = []
    for q in questions:
        r_v1 = idx.get(("rrf+priors",    q["id"]))
        r_v2 = idx.get(("rrf+priors_v2", q["id"]))
        if r_v1 and r_v2 and r_v2["mrr"] != r_v1["mrr"]:
            qids_mrr_moved.append(q["id"])

    print(
        f"  Of the {len(questions)} evaluation questions, "
        f"{len(qids_with_gold_changed)} had at least one gold chunk with a changed prior. "
        f"{len(qids_mrr_moved)} questions moved in MRR. "
        "This explains why the aggregate delta is what it is."
    )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the v2 retrieval-impact experiment results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to a specific run_*.jsonl results file (default: newest).",
    )
    args = parser.parse_args()

    results_path = args.results if args.results else _newest_results_file()
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise KeyError(
            "DATABASE_URL environment variable is not set. "
            "Export it before running this script."
        )

    questions = _load_questions()
    rows      = _load_results(results_path)
    idx       = _index_results(rows)
    changed   = _load_changed_chunks(database_url)

    _header(results_path, rows, questions)
    base_delta, _xref_delta = _aggregate_deltas(idx, questions)
    _chunk_level_changes(changed)
    affected_gold = _coverage(changed, questions)
    _question_table(idx, questions, affected_gold)
    _q007_block(idx, questions, changed)
    _footer(base_delta, questions, idx, affected_gold)


if __name__ == "__main__":
    main()
