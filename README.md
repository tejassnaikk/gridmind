# GridMind

GridMind is a retrieval and evaluation system for NERC CIP cybersecurity compliance documents. It answers a causal question most RAG projects skip: does improving obligation classification measurably improve downstream retrieval?

## Why this project is different

Most RAG portfolios report retrieval or LLM metrics and stop. GridMind runs a pre-registered controlled experiment measuring whether a learned obligation classifier's output, when substituted for regex-derived retrieval priors, changes downstream ranking. The architecture has two stages: a classifier assigns obligation labels to corpus chunks, those labels feed into multiplicative priors applied after Reciprocal Rank Fusion, and the ranked output is measured by Recall@5 and MRR. The experiment result is measured, small (+0.038 MRR), and honestly explained — including the finding that the improvement came from an indirect mechanism the pre-registration did not predict.

## Key contributions

- Built a hybrid RRF retrieval pipeline (dense bge-small-en-v1.5 + Postgres FTS) over 96 chunks from CIP-002-5.1a, CIP-013-2, and CIP-013-3, with freshness and obligation-strength priors applied as multipliers after fusion.
- Hand-labeled 96 corpus chunks with a documented 9-rule taxonomy (see `classifier/labels/RULES.md`) and trained four obligation classifiers evaluated by 5-fold stratified out-of-fold cross-validation.
- Improved obligation classification from regex F1=0.727 to Logistic Regression F1=0.913 under a controlled protocol (same folds, same seed, same 96 labels).
- Designed and executed a pre-registered retrieval A/B experiment measuring how learned obligation classification affects downstream ranking, yielding +0.038 MRR and explaining the underlying retrieval mechanism.
- Identified a persistent representational blind spot across three model families: exemption language expressed as asset descriptions remained difficult despite changes in architecture.
- Documented negative and null results alongside positive ones, including a mode-collapsed DistilBERT ablation and the failure of the pre-registered q007 rescue.

## Results at a glance

**Classification** (5-fold stratified OOF-CV, n=96, 23 positives):

| Model | F1 |
|---|---|
| Regex baseline | 0.727 |
| **Logistic Regression** | **0.913** |
| DistilBERT v1 (default loss) | 0.516 |
| DistilBERT v2 (class-weighted loss) | 0.792 |

**Retrieval** (10 questions, top-K=5):

| Experiment | Result |
|---|---|
| Baseline Recall@5 (RRF + regex priors) | 0.750 |
| Baseline MRR | 0.545 |
| **Classifier-driven priors (LogReg OOF)** | **+0.038 MRR** |

## Architecture

```mermaid
flowchart LR
    PDF[PDF Documents] --> Parse[Parse + Chunk]
    Parse --> RegexTag[Regex Obligation Tag]
    RegexTag --> Embed[bge-small-en-v1.5<br>Embedding]
    Embed --> PG[(Postgres<br>pgvector + FTS)]
    Query[User Query] --> QEmbed[Query Embedding]
    QEmbed --> Dense[Dense Retrieval<br>cosine top-40]
    Query --> Sparse[Sparse Retrieval<br>ts_rank top-40]
    Dense --> RRF[RRF Fusion]
    Sparse --> RRF
    RRF --> Priors[Priors Application<br>freshness × obligation]
    Priors --> TopK[Top-K Output]
    PG --> Dense
    PG --> Sparse
```

*Alternate priors source: the `obligation_strength_v2` column populated by the LogReg classifier (see `db/migrations/002_populate_obligation_strength_v2.py`) can be swapped in for A/B comparison without changing any retrieval code.*

At ingestion time, PDFs are parsed and split into requirement-aligned chunks by `ingestion/chunk.py`, tagged with regex-derived obligation labels, embedded by `ingestion/embed.py`, and written to Postgres with pgvector indexes and a `body_tsvector` column for full-text search. At query time, `retrieval/retrievers.py` runs dense and sparse searches independently, `retrieval/scorer.py` fuses their ranked lists by Reciprocal Rank Fusion, and freshness and obligation priors are applied as multipliers to produce the final ranking.

## Classifier progression

The regex classifier was the natural starting point: one `re.compile` call per modal verb, F1=0.727, fully interpretable, no training data required. It became the floor to beat and the source of `regex_label` in the labeled dataset. Its failure modes were well-defined: it missed 7 obligation-bearing chunks (Measures and applicability sections that use binding language without the word `shall`) and over-fired on 5 boilerplate chunks where `shall` governs CEA administrative roles rather than Responsible Entity obligations.

Logistic Regression on TF-IDF bigrams (1–2 grams, `min_df=2`) with `class_weight="balanced"` reached F1=0.913 in 5-fold OOF-CV. The gain came from both directions: the classifier learned to recognize obligation-bearing Measures (`M1.`, `M2.`) and applicability sections that the regex missed, and it suppressed the boilerplate false positives. The per-fold F1 scores were stable (`fold1=0.889, fold2=0.800, fold3=1.000, fold4=1.000, fold5=0.889`), which matters on a 96-row dataset where one fold's composition can swing aggregate metrics.

DistilBERT v1 used default cross-entropy loss with `num_train_epochs=3`, `learning_rate=2e-5`, `per_device_train_batch_size=8`. With 76 training examples per fold and a 4:1 class imbalance, the model collapsed to predicting the majority class on most folds: F1=0.516. This is an ablation result: two folds collapsed to F1=0.000 (majority-class prediction only), and the aggregate breakdown was TP=8, FP=0, FN=15, TN=73 across the 96 chunks.

DistilBERT v2 is identical to v1 except for a `WeightedTrainer` subclass that replaces the default loss with `CrossEntropyLoss(weight=[1.0, n_neg/n_pos])` computed per training fold. F1 rose to 0.792, recovering most of the gap to LogReg by changing one hyperparameter. The remaining gap reflects the small training set and the fact that pretraining on general text does not give DistilBERT much advantage over bigram TF-IDF when the vocabulary is domain-specific and the training set has fewer than 100 examples.

LogReg wins on aggregate F1 and provides the `obligation_strength_v2` column for the retrieval-impact experiment. The dataset has n=96 with 23 positives. Further classification gains likely require corpus expansion, not increased model complexity.

## The retrieval-impact experiment

### Question

Does substituting the classifier's out-of-fold predictions for regex-derived obligation priors change downstream retrieval quality?

### Design

Out-of-fold predictions were used to prevent leakage: each chunk's predicted label was generated by a model trained on the other 76 chunks in its fold, so the label for a given chunk was never influenced by a model that had seen that chunk. The 5-fold assignments used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical to the LogReg baseline, so fold membership is byte-identical across both scripts. The mapping from (regex label, classifier prediction) to `obligation_strength_v2` was locked before running the eval:

| regex_label | classifier | obligation_strength_v2 |
|---|---|---|
| shall | 1 | shall |
| should | 1 | should |
| may | 1 | may |
| informational | 1 | should (default obligation weight) |
| any | 0 | informational |

The `should` assigned for informational-to-obligation promotions is a retrieval-weight mapping, not a semantic claim. The classifier produces binary output, so an intermediate weight is assigned when it detects obligation force without an explicit modal verb. The `obligation_strength_v2` column is a shadow column added by `db/migrations/002_add_obligation_strength_v2.sql`, leaving the original `obligation_strength` column untouched. Re-running `classifier/export_oof_predictions.py` asserts F1=0.913 to three decimal places as its reproducibility contract.

### Result

ΔR@5 = 0.000. ΔMRR = +0.038. The 33 changed chunks broke down as: 21 `should`/`may` chunks demoted to `informational`, 7 `informational` chunks promoted to `should`, and 5 `shall` chunks demoted. Of the 9 unique gold chunks in the evaluation set, only 2 had their prior changed.

### Mechanism

The +0.038 MRR gain came primarily from distractor demotion, not from the pre-registered mechanism of rescuing q007. Questions q006 and q008 gained +0.500 MRR and +0.133 MRR respectively despite having no gold-side prior changes. In both cases, non-gold chunks competing for top positions had their obligation weights demoted from `should` or `may` to `informational`, allowing gold chunks to climb without any direct boost. The pre-registered rescue of q007 did not occur: the promoted gold chunk (`03ef96af`) remained outside the top-10 under both v1 and v2 configurations. Question q010 showed a net negative: `e9d5ad38` was demoted from `should` to `informational` (a classifier error on Background commentary), dropping MRR from 0.500 to 0.250.

### Interpretation

The classifier improved retrieval, but not for the reason predicted. Binary obligation classification deployed as a retrieval prior can help indirectly by suppressing noise even when it cannot help directly by boosting the right chunks. This is a genuine finding about the mechanism by which classifier corrections interact with multiplicative priors.

## Persistent representational blind spot

Three model families fail on the same two chunks. In the retrieval eval, bge-small ranks `466235f7` and `e83242ae` outside the top-10 for q007 in both v1 and v2 configurations. In the classifier eval, LogReg and DistilBERT v2 both misclassify these chunks as false negatives. Both chunks encode exemption scope for CIP-013: "Cyber Assets associated with communication networks and data communication links."

The failure mode is consistent: exemption scope is expressed as asset-class description rather than requirement language. This produces a semantic gap that multiplicative priors cannot close (the weight ratio between `should` and `informational` is 1.15, which is not enough to move a chunk from rank 20 to rank 5) and classifier retraining on the same representations does not resolve. The fix is likely representational: query rewriting to explicitly reference exemption language, embedding fine-tuning on regulatory exemption patterns, or cross-encoder reranking that can attend to the full query-chunk pair.

## Limitations

- Evaluation uses 10 questions. The +0.038 MRR result is measured on a small benchmark; statistical robustness requires expansion to 30–50 questions, which is the planned next step.
- Binary obligation labeling loses modality information (`shall` / `should` / `may`) that the regex captured cheaply. The classifier's demotion of every `should` and `may` chunk to `informational` reflects the labeling policy, not a bug. It is the primary mechanism behind the 21 demotions.
- The pre-registered q007 rescue did not occur; the observed MRR gain came from an indirect mechanism.
- The corpus is 96 chunks across three related NERC CIP standards. Generalization to broader regulatory corpora is unmeasured.
- The chunker has a documented bug affecting three Interpretation Q&A chunks (see the TODO in `ingestion/chunk.py`). Downstream code treats `requirement_id` as advisory to work around it.

## Reproducing the results

```bash
cd gridmind
conda activate gridmind
docker start gridmind-pg
export DATABASE_URL="postgresql://gridmind:gridmind@localhost:5432/gridmind_embeddings"
```

Regenerate LogReg OOF predictions (asserts F1=0.913):

```bash
python -m classifier.export_oof_predictions
```

Apply the v2 migrations:

```bash
docker exec -i gridmind-pg psql -U gridmind -d gridmind_embeddings \
    < db/migrations/002_add_obligation_strength_v2.sql
python -m db.migrations.002_populate_obligation_strength_v2
```

Run the full eval (all 7 strategies, 10 questions):

```bash
python -m eval.run
```

Generate the A/B analysis report:

```bash
python -m eval.analyze_v2_impact
```

Reproducibility contracts: `classifier/export_oof_predictions.py` asserts OOF F1=0.913 before writing output. The baseline `rrf+priors` strategy reproduces R@5=0.750 and MRR=0.545 byte-identically across runs because the retrieval config, model weights, and fold assignments are all fixed.

## Technology stack

- Python 3.11
- PyTorch 2.12
- transformers 5.12 (Hugging Face)
- sentence-transformers with BAAI/bge-small-en-v1.5 (384-dim, 512-token window)
- scikit-learn 1.9 (TF-IDF, LogisticRegression, StratifiedKFold)
- Postgres 16 with pgvector extension
- FastAPI (HTTP query layer at `api/main.py`)
- psycopg 3 with psycopg-pool
- Docker (containerized Postgres)

## Repository layout

```
gridmind/
├── ingestion/      Parse PDFs, chunk by requirement boundary, classify obligation strength, embed, upsert to Postgres
├── retrieval/      Dense + sparse retrievers, RRF fusion, prior application, cross-reference expansion
├── classifier/     Labeling pipeline, training scripts, OOF prediction export, baselines
├── db/             Schema SQL, migrations (DDL + population scripts)
├── eval/           Gold questions, IR metrics, strategy functions, eval runner, v2 impact analysis
├── api/            FastAPI HTTP layer with connection pooling
└── tests/          Unit tests for scorer (no DB required)
```

## Roadmap

- Expand the evaluation benchmark to 30–50 questions (frozen corpus, frozen retrieval config, frozen classifier) to test whether the +0.038 MRR result is robust to sampling.
- Investigate embedding fine-tuning or cross-encoder reranking to address the exemption-language representational blind spot identified in q007.
- Add an answer synthesis layer with groundedness and citation evaluation, deferred until the retrieval story is statistically firmer.
