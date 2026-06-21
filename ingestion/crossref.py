"""
Regex-based NERC standard cross-reference extraction.

Matches citations of the form  FAM-NNN  or  FAM-NNN-V  where:
  FAM  — two or three uppercase letters (family code, e.g. CIP, FAC, PRC)
  NNN  — three digits (standard number)
  -V   — optional version suffix (one or more digits)

Examples matched:  CIP-013-2  FAC-001-3  PRC-005-6  TOP-001  NUC-001-3

Self-reference filtering (dropping the current standard's own id) is
intentionally deferred to the upsert stage, not done here.
"""

from __future__ import annotations

import re

# Anchored at word boundaries so we don't partially match longer tokens.
_CITATION = re.compile(r"\b([A-Z]{2,3}-\d{3}(?:-\d+)?)\b")


def extract_related_standards(body: str) -> list[str]:
    """Return a sorted, de-duplicated list of NERC standard citations in *body*."""
    return sorted(set(_CITATION.findall(body)))
