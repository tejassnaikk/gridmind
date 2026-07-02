-- Migration 002: add shadow column obligation_strength_v2 to standard_chunks
-- Purpose:       shadow column for retrieval-impact A/B experiment comparing
--                LogReg-classifier-driven priors against the regex baseline
-- Idempotent:    ADD COLUMN IF NOT EXISTS and CREATE INDEX IF NOT EXISTS;
--                safe to re-run against an already-migrated schema

ALTER TABLE standard_chunks
    ADD COLUMN IF NOT EXISTS obligation_strength_v2 TEXT
        CHECK (obligation_strength_v2 IN ('shall', 'should', 'may', 'informational'));

COMMENT ON COLUMN standard_chunks.obligation_strength_v2 IS
    'LogReg-classifier-driven obligation label (v2). '
    'Values derived from the OOF predictions in '
    'classifier/predictions/logreg_oof_v1.tsv via the mapping: '
    'oof_pred=1 → carry regex_label (shall/should/may) or ''should'' for informational; '
    'oof_pred=0 → ''informational''. '
    'Populated by db/migrations/002_populate_obligation_strength_v2.py.';

CREATE INDEX IF NOT EXISTS idx_chunks_obligation_strength_v2
    ON standard_chunks (obligation_strength_v2);
