"""
GridMind retrieval eval runner.

Usage:
    python -m eval.run

Runs all five strategies against the 10 gold questions, prints a summary
table, and writes per-(strategy, question) results to eval/results/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from eval.metrics import recall_at_k, reciprocal_rank
from eval.strategies import STRATEGIES
from retrieval.retrievers import register_vector

load_dotenv()

_EVAL_DIR = Path(__file__).parent
_QUESTIONS_FILE = _EVAL_DIR / "questions.jsonl"
_RESULTS_DIR = _EVAL_DIR / "results"

_RETRIEVE_K = 10   # over-retrieve; metrics computed at @5 and @10
_COL_W = 22        # strategy column width for the summary table


def _load_questions() -> list[dict]:
    questions = []
    with open(_QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main() -> None:
    questions = _load_questions()

    database_url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(database_url)
    register_vector(conn)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _RESULTS_DIR.mkdir(exist_ok=True)
    results_path = _RESULTS_DIR / f"run_{timestamp}.jsonl"

    # ------------------------------------------------------------------ #
    # Run all strategies × all questions
    # ------------------------------------------------------------------ #
    # summary_data[strategy] = list of (r@5, r@10, mrr) per question
    summary_data: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in STRATEGIES
    }

    with open(results_path, "w") as results_file:
        for strategy_name, strategy_fn in STRATEGIES.items():
            print(f"  running {strategy_name} ...", flush=True)
            for q in questions:
                retrieved = strategy_fn(q["question"], conn, _RETRIEVE_K)
                gold = q["gold_chunk_ids"]

                r5  = recall_at_k(retrieved, gold, 5)
                r10 = recall_at_k(retrieved, gold, 10)
                mrr = reciprocal_rank(retrieved, gold)

                summary_data[strategy_name].append((r5, r10, mrr))

                row = {
                    "timestamp": timestamp,
                    "strategy": strategy_name,
                    "question_id": q["id"],
                    "retrieved_ids": retrieved,
                    "gold_ids": gold,
                    "recall_at_5": r5,
                    "recall_at_10": r10,
                    "mrr": mrr,
                }
                results_file.write(json.dumps(row) + "\n")

    conn.close()

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    header = (
        f"{'Strategy':<{_COL_W}}  {'Recall@5':>9}  {'Recall@10':>10}  {'MRR':>7}"
    )
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)
    for strategy_name, scores in summary_data.items():
        mean_r5  = sum(s[0] for s in scores) / len(scores)
        mean_r10 = sum(s[1] for s in scores) / len(scores)
        mean_mrr = sum(s[2] for s in scores) / len(scores)
        print(
            f"{strategy_name:<{_COL_W}}  {mean_r5:>9.3f}  {mean_r10:>10.3f}  {mean_mrr:>7.3f}"
        )

    print()
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
