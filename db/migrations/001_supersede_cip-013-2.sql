-- 001_supersede_cip-013-2.sql
--
-- One-off data migration: mark CIP-013-2 as superseded by CIP-013-3.
--
-- Context:
--   GridMind's freshness prior (retrieval/scorer.py) reads
--   standard_document_metadata.superseded_by. When NULL, the document
--   is treated as current (freshness=1.0). When set to a successor
--   standard id, the document is downweighted (freshness=0.5).
--
--   NERC published CIP-013-3 with conforming changes for virtualization
--   (Project 2016-02), superseding CIP-013-2. Until an ingestion-time
--   process derives this from NERC's standard registry, supersede links
--   are set manually via migration files like this one.
--
-- Idempotency:
--   IS DISTINCT FROM handles NULL safely; re-running this script is a
--   no-op once the link is set. updated_at only moves on real change.
--
-- Verification:
--   After running, the SELECT at the bottom should show:
--     CIP-013 v2 -> superseded_by = 'CIP-013-3'
--     CIP-013 v3 -> superseded_by = NULL

BEGIN;

UPDATE standard_document_metadata
SET    superseded_by = 'CIP-013-3',
       updated_at    = NOW()
WHERE  standard_id   = 'CIP-013'
  AND  version       = 2
  AND  superseded_by IS DISTINCT FROM 'CIP-013-3';

COMMIT;

SELECT standard_id, version, superseded_by
FROM   standard_document_metadata
WHERE  standard_id = 'CIP-013'
ORDER  BY version;
