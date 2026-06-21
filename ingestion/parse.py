"""
PDF text extraction for NERC/FERC standards.

extract_pages returns one dict per page.  Running headers and footers that
repeat verbatim across many pages are stripped, but the text is otherwise
left intact — over-cleaning before chunking discards structural cues we need.
"""

from __future__ import annotations

import re
from collections import Counter

import fitz  # PyMuPDF


# Minimum fraction of pages a line must appear on to be called a
# header/footer.  0.40 catches most NERC boilerplate without touching
# section titles that happen to repeat a few times.
_REPEAT_THRESHOLD = 0.40

# "Page 1 of 10", "page 12 of 47", etc. — varies per page so the
# frequency filter misses them.
_PAGE_OF_PAGE = re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE)


def _candidate_lines(doc: fitz.Document) -> set[str]:
    """Return lines that appear on >= REPEAT_THRESHOLD of all pages."""
    n_pages = len(doc)
    if n_pages == 0:
        return set()

    counter: Counter[str] = Counter()
    for page in doc:
        seen_on_page: set[str] = set()
        for line in page.get_text().splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen_on_page:
                counter[stripped] += 1
                seen_on_page.add(stripped)

    cutoff = max(2, int(n_pages * _REPEAT_THRESHOLD))
    return {line for line, count in counter.items() if count >= cutoff}


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Open *pdf_path* and return one dict per page::

        {"page_number": int,   # 1-based
         "text":       str}    # cleaned page text
    """
    doc = fitz.open(pdf_path)
    boilerplate = _candidate_lines(doc)

    pages: list[dict] = []
    for page in doc:
        lines = page.get_text().splitlines()
        kept = [
            l for l in lines
            if l.strip() not in boilerplate
            and not _PAGE_OF_PAGE.search(l)
        ]
        text = "\n".join(kept).strip()
        pages.append({"page_number": page.number + 1, "text": text})

    doc.close()
    return pages
