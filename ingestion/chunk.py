"""
Section-aware chunking for NERC reliability standards.

NERC structure targeted:
  R1.  <requirement text>
       1.1. <sub-requirement>
       1.2. <sub-requirement>
  M1.  <measure text>
  R2.  ...

Strategy
--------
Walk the full text line-by-line.  A line that starts a new top-level
Requirement (R1., R2., …) or Measure (M1., M2., …) flushes the current
chunk and opens a new one tagged with that requirement id.

Lettered section headers (A. Introduction … E. Associated Documents) and
"Version History" also flush the accumulator and open a new chunk with
requirement_id=None so that trailing compliance / variance / history
content is never appended to the last Measure.

Text before the first Requirement becomes preamble.  If the preamble
exceeds PREAMBLE_CHAR_LIMIT it is split into smaller chunks so the
retriever never receives an unusably large block.

Each returned dict:
    chunk_index    int   — 0-based sequential
    requirement_id str|None — "R1", "M1", None for preamble/section
    body           str   — full text of this chunk
    page_number    int   — page where the chunk STARTS (1-based)
"""

from __future__ import annotations

import re

# Regex for a top-level requirement or measure heading, e.g.:
#   "R1."  "R12."  "M1."  "M12."
# The pattern matches at the start of a stripped line, optionally preceded
# by whitespace.  We do NOT match sub-requirements like "1.1." here —
# those stay inside their parent chunk.
_REQ_START = re.compile(r"^(R\d+|M\d+)\.\s")

# Lettered NERC section headers: "A. Introduction", "B. Requirements and
# Measures", "C. Compliance", "D. Regional Variances", "E. Associated
# Documents".  Guard: stripped line must be <= 60 chars so we never split
# on a prose sentence that opens with a capital letter and a period.
_SECTION_LETTER = re.compile(r"^[A-E]\.\s+\S", re.IGNORECASE)
_SECTION_MAX_LEN = 60

# Version History table header — appears at the end of most NERC standards.
_VERSION_HISTORY = re.compile(r"^version\s+history", re.IGNORECASE)

# Hard cap applied to every chunk in the post-processing pass.
# bge-small-en-v1.5 has a 512-token window; 1800 chars ≈ 360 tokens for
# typical regulatory English, leaving headroom for augmented_text context.
MAX_CHARS = 1_800

# Preamble pre-split limit — a rough first pass; the post-processing cap
# (MAX_CHARS) is the authoritative final constraint.
PREAMBLE_CHAR_LIMIT = 3_000


def _flush(
    chunks: list[dict],
    body_lines: list[str],
    req_id: str | None,
    start_page: int,
) -> None:
    body = "\n".join(body_lines).strip()
    if body:
        chunks.append(
            {
                "chunk_index": len(chunks),
                "requirement_id": req_id,
                "body": body,
                "page_number": start_page,
            }
        )


def _split_preamble(text: str, start_page: int) -> list[dict]:
    """Break oversized preamble text at blank lines into sub-chunks."""
    chunks: list[dict] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > PREAMBLE_CHAR_LIMIT and current_parts:
            chunks.append(
                {
                    "chunk_index": 0,  # renumbered by caller
                    "requirement_id": None,
                    "body": "\n\n".join(current_parts),
                    "page_number": start_page,
                }
            )
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append(
            {
                "chunk_index": 0,
                "requirement_id": None,
                "body": "\n\n".join(current_parts),
                "page_number": start_page,
            }
        )
    return chunks


def _split_oversized(chunk: dict) -> list[dict]:
    """
    Sub-split a single chunk that exceeds MAX_CHARS.

    Algorithm:
      1. Split body on blank lines (paragraph boundaries).
      2. Greedily pack paragraphs into pieces ≤ MAX_CHARS.
      3. If one paragraph alone exceeds MAX_CHARS, split it further on
         sentence boundaries (period / exclamation / question + whitespace).

    Sub-chunks inherit requirement_id and page_number; chunk_index is
    placeholder 0 and renumbered by the caller.
    """
    if len(chunk["body"]) <= MAX_CHARS:
        return [chunk]

    req_id = chunk["requirement_id"]
    page = chunk["page_number"]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk["body"]) if p.strip()]

    pieces: list[str] = []
    buf_parts: list[str] = []
    buf_len = 0

    def _flush_buf() -> None:
        if buf_parts:
            pieces.append("\n\n".join(buf_parts))
        buf_parts.clear()
        nonlocal buf_len
        buf_len = 0

    for para in paragraphs:
        if len(para) > MAX_CHARS:
            # Flush whatever is buffered before dealing with this giant paragraph
            _flush_buf()
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf: list[str] = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) > MAX_CHARS and sent_buf:
                    pieces.append(" ".join(sent_buf))
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(sent)
                sent_len += len(sent) + 1  # +1 for the space between sentences
            if sent_buf:
                pieces.append(" ".join(sent_buf))
        else:
            if buf_len + len(para) > MAX_CHARS:
                _flush_buf()
            buf_parts.append(para)
            buf_len += len(para)

    _flush_buf()

    return [
        {
            "chunk_index": 0,  # renumbered by caller
            "requirement_id": req_id,
            "body": piece,
            "page_number": page,
        }
        for piece in pieces
        if piece.strip()
    ]


def chunk_standard(pages: list[dict]) -> list[dict]:
    """
    Chunk a NERC standard into requirement-aligned segments.

    Each top-level Requirement (R1., R2., …) and Measure (M1., M2., …)
    becomes one chunk together with all its sub-requirements and prose.
    Text before the first Requirement becomes one or more preamble chunks
    (requirement_id=None).
    """
    # ------------------------------------------------------------------ #
    # Pass 1: walk every line and tag it with (page_number, line_text)
    # ------------------------------------------------------------------ #
    tagged_lines: list[tuple[int, str]] = []
    for page in pages:
        for line in page["text"].splitlines():
            tagged_lines.append((page["page_number"], line))

    # ------------------------------------------------------------------ #
    # Pass 2: split on top-level Requirement / Measure boundaries
    # ------------------------------------------------------------------ #
    raw_chunks: list[dict] = []          # before final index assignment
    current_lines: list[str] = []
    current_req: str | None = None
    current_page: int = pages[0]["page_number"] if pages else 1
    in_preamble = True

    for page_num, line in tagged_lines:
        stripped = line.strip()
        req_match = _REQ_START.match(stripped)
        section_match = (
            not req_match
            and len(stripped) <= _SECTION_MAX_LEN
            and (
                _SECTION_LETTER.match(stripped)
                or _VERSION_HISTORY.match(stripped)
            )
        )

        if req_match or section_match:
            # Flush whatever we have accumulated
            if in_preamble:
                preamble_text = "\n".join(current_lines).strip()
                if preamble_text:
                    raw_chunks.extend(
                        _split_preamble(preamble_text, current_page)
                    )
                in_preamble = False
            else:
                _flush(raw_chunks, current_lines, current_req, current_page)

            current_req = req_match.group(1) if req_match else None
            current_lines = [line]
            current_page = page_num
        else:
            # Track the page only when starting a new logical section
            if not current_lines:
                current_page = page_num
            current_lines.append(line)

    # Flush the final chunk
    if in_preamble:
        preamble_text = "\n".join(current_lines).strip()
        if preamble_text:
            raw_chunks.extend(_split_preamble(preamble_text, current_page))
    else:
        _flush(raw_chunks, current_lines, current_req, current_page)

    # ------------------------------------------------------------------ #
    # Pass 3: cap oversized chunks at MAX_CHARS
    # ------------------------------------------------------------------ #
    capped: list[dict] = []
    for chunk in raw_chunks:
        capped.extend(_split_oversized(chunk))

    # ------------------------------------------------------------------ #
    # Pass 4: assign sequential chunk_index across the whole document
    # ------------------------------------------------------------------ #
    for i, chunk in enumerate(capped):
        chunk["chunk_index"] = i

    return capped
