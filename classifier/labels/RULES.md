# GridMind Obligation Labeling Rules — v1

Binary classification: does a chunk impose a mandatory action,
prohibition, scope definition, or evidence requirement on a Responsible
Entity?

`1` = YES (obligation). `0` = NO (everything else).

Applied to: `classifier/labels/to_label.tsv` → `classifier/labels/labeled_v1.tsv`
Produced by: single annotator, applying the rules below mechanically.

Distribution: 23 obligations, 73 non-obligations across 96 chunks.
Disagreement with regex baseline: 12 chunks (~12.5%) — 7 regex
under-tags + 5 regex over-tags. This is the signal the classifier is
trained to capture.

## Decision tree

Apply rules in strict order. Stop at the first rule that fires.

### Rule 1 — Numbered Requirements and Measures (always 1)

If the chunk body IS a numbered requirement or measure (R1, R2, R3, M1,
M2, M3) — meaning the chunk text itself states the obligation or
evidence clause — label = `1`.

The `req_id` metadata column is a CHUNKER OUTPUT, not evidence. If the
chunker tags a chunk with `req_id=R1` but the body is interpretation,
guidance, or commentary about R1 rather than R1 itself, this rule does
NOT fire. Fall through to Rule 6 (Interpretation Q&A) or Rule 8
(Guidelines) as appropriate.

Spot the difference:
- `"R1. Each Responsible Entity shall implement..."` → Rule 1 fires → `1`
- `"Energy Sector Security Consortium submitted a Request for
   Interpretation seeking clarification of R1..."` → Rule 6, not Rule 1 → `0`

Content beats metadata when they conflict.

### Rule 2 — Applicability / Facilities / Exemptions sections (always 1)

If the chunk body contains text from section **4.x** of a standard —
specifically 4.1 (Functional Entities), 4.2 (Facilities), or 4.2.3
(Exemptions) — label = `1`.

These define mandatory scope. They don't always say `shall` but they
are operationally binding ("the standard applies to X / does not apply
to Y").

How to spot: body contains substrings like `4.1.`, `4.2.`, `4.2.1`,
`4.2.3.`, `Facilities:`, `Exemptions:`, `Functional Entities:`, or a
list of entity types (Balancing Authority, Distribution Provider,
Generator Operator, etc.).

### Rule 3 — Compliance section with an action obligation (1)

A chunk in the **C. Compliance** section is `1` only if it contains a
`shall` clause directing the Responsible Entity to DO something
specific (retain evidence, comply with a process, submit information).

Examples:
- `1`: `"The Responsible Entity shall keep data or evidence..."`
- `1`: `"Each Responsible Entity shall retain evidence... for three calendar years"`

Otherwise, the chunk is `0` even if `shall` appears (see Rule 4).

### Rule 4 — Compliance definitions, CEA prose, VSL tables (always 0)

The following Compliance-section content is `0` even when `shall` appears:
- Definitions like `"'Compliance Enforcement Authority' (CEA) means..."`
- CEA role descriptions (`"The Regional Entity shall serve as the CEA..."`)
- All Violation Severity Level (VSL) tables
- Compliance Monitoring Process descriptions

### Rule 5 — Attachment 1 / Impact Criteria (always 0)

Attachment 1 of CIP-002 explicitly states: *"the criteria defined in
Attachment 1 do not constitute stand-alone compliance requirements."*

Any chunk whose body is from Attachment 1's impact-rating tables
(1. High Impact Rating, 2. Medium Impact Rating, 3. Low Impact Rating,
criteria like 1.1, 2.1, 2.6, etc.) is `0`.

### Rule 6 — Guidelines, Technical Basis, Rationale, Interpretation Q&A (always 0)

Explanatory prose is `0`. Includes:
- "Guidelines and Technical Basis" sections
- "Rationale:" sections
- Interpretation Q&A
- Discussion of why a criterion exists or what an example illustrates

Even when these chunks quote `shall`, the quote is commentary. The
chunk is not itself an obligation.

### Rule 7 — Headers, metadata, document furniture (always 0)

- Section headers alone ("B. Requirements and Measures")
- "A. Introduction", "Title:", "Number:", "Purpose:" front matter
- "D. Regional Variances: None", "E. Interpretations: None"
- "Effective Date:" sections
- Version History tables
- Diagram captions, X-matrices, figure prose

### Rule 8 — Operations Service guidelines (always 0)

CIP-002 Guidelines chunks describing operational services (Balancing
Load and Generation, Controlling Voltage, Restoration of BES, Dynamic
Response). Describe what these services entail, not what an RE must do.

### Rule 9 — Background / preamble (always 0)

The "Background:" section of CIP-002 and similar preamble sections that
introduce concepts without imposing obligations are `0`.

## Tiebreaker

If a chunk could fall under multiple rules, the earlier rule wins.
EXCEPTION: Rule 1's "content beats metadata" clause overrides the
ordering when the only Rule-1 signal is misleading `req_id` metadata.

## Provenance

Labels in `labeled_v1.tsv` were produced by a single annotator
following this decision tree, with chunk_id-keyed merge via
`classifier/merge_labels.py`. The original regex labels remain in the
`regex_label` column for comparison.

A future revision should be hand-verified by a second annotator on the
disagreement set (currently 12 chunks) before being used as a held-out
test set for a published benchmark.
