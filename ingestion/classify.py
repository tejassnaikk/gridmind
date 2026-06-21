"""
Regex-based obligation classification for NERC standard chunks.

Returns one of the four tags used by the retrieval scoring priors:
  shall          — mandatory obligation  (prior multiplier 1.00)
  should         — strong recommendation (prior multiplier 0.92)
  may            — permissive            (prior multiplier 0.85)
  informational  — no obligation keyword (prior multiplier 0.80)

Precedence: shall > should > may.  A chunk containing both "shall" and
"may" (e.g. an R with an embedded definition) is tagged "shall".  This
is intentionally coarse — the DistilBERT classifier in week 3 will
replace it for requirement bodies.
"""

from __future__ import annotations

import re

_SHALL = re.compile(r"\bshall\b", re.IGNORECASE)
_SHOULD = re.compile(r"\bshould\b", re.IGNORECASE)
_MAY = re.compile(r"\bmay\b", re.IGNORECASE)


def classify_obligation(body: str) -> str:
    """Return 'shall' | 'should' | 'may' | 'informational'."""
    if _SHALL.search(body):
        return "shall"
    if _SHOULD.search(body):
        return "should"
    if _MAY.search(body):
        return "may"
    return "informational"
