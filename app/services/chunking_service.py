"""Split parsed Markdown into overlapping chunks for embedding.

Chunking is the biggest lever on retrieval quality after the embedding model:
too large and a chunk dilutes its own meaning (the embedding averages several
topics, so nothing matches sharply); too small and it loses the context that
makes it answerable. The target here is ~2000 characters per chunk with ~200
characters of overlap, so a passage split across a boundary still appears whole
in one of the two neighbouring chunks.

Two rules shape the algorithm:

* **Respect paragraph boundaries.** Markdown separates blocks (paragraphs,
  headings, list items) with blank lines. Packing whole blocks keeps a chunk
  from starting or ending mid-sentence, which is where embeddings get muddy.
* **Overlap by whole paragraphs where possible.** When a chunk fills up, the
  next one is seeded with the trailing paragraph(s) of the previous chunk (up to
  the overlap budget), so continuity is preserved without cutting text.

A single paragraph larger than the chunk size (a giant table row, or prose with
no blank lines) can't be split on a boundary, so it falls back to a fixed
character window — still overlapping — as a last resort.

Pure and synchronous: no I/O, so it's trivial to unit-test and cheap to call
inline from the ingestion pipeline.
"""

import re

# Library defaults for standalone/test use. The application overrides these per
# request from config (settings.chunk_size / chunk_overlap → passed by
# ingestion_service), so tuning in production is a .env change, not a code edit.
# Targets, not hard physics: a chunk never *exceeds* chunk_size, but may be
# shorter when a paragraph boundary falls early.
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200

# One or more blank lines (blank = nothing but whitespace) separate Markdown
# blocks. A single newline inside a block (a soft wrap) is left intact.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> list[str]:
    """Break Markdown into stripped, non-empty blocks on blank lines."""
    return [block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip()]


def _joined_len(paragraphs: list[str]) -> int:
    """Length of ``"\\n\\n".join(paragraphs)`` without building the string."""
    if not paragraphs:
        return 0
    return sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a too-long block into overlapping fixed-size character windows.

    The fallback for a single paragraph bigger than ``chunk_size``: there is no
    boundary to break on, so the window slides by ``chunk_size - overlap`` and
    each piece shares ``overlap`` characters with the next.
    """
    step = chunk_size - overlap  # > 0: caller guarantees overlap < chunk_size
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return pieces


def _overlap_tail(paragraphs: list[str], overlap: int) -> list[str]:
    """The longest suffix of ``paragraphs`` whose joined length is <= ``overlap``.

    Seeds the next chunk with as much trailing context as the overlap budget
    allows, kept to whole paragraphs. May be empty if even the last paragraph
    alone exceeds the budget (then that boundary carries no paragraph overlap —
    a hard-split block, already self-overlapping, is the usual reason).
    """
    tail: list[str] = []
    for paragraph in reversed(paragraphs):
        candidate_len = len(paragraph) + (2 if tail else 0) + _joined_len(tail)
        if candidate_len > overlap:
            break
        tail.insert(0, paragraph)
    return tail


def chunk_markdown(
    markdown: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split Markdown into overlapping chunks, each at most ``chunk_size`` chars.

    Returns chunks in document order. Empty or whitespace-only input yields ``[]``
    (the ingestion pipeline already treats no-text as a failure upstream, but this
    stays total rather than assuming that).
    """
    if overlap >= chunk_size:
        # Otherwise the window can't advance (step <= 0) and overlap is
        # meaningless — a caller error worth failing loudly on.
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    # Expand any oversized paragraph into hard-split units first, so the packer
    # below only ever handles units that fit in a chunk.
    units: list[str] = []
    for paragraph in _split_paragraphs(markdown):
        if len(paragraph) > chunk_size:
            units.extend(_hard_split(paragraph, chunk_size, overlap))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        # Would adding this unit overflow the current chunk? Emit and reset first.
        if current and _joined_len(current) + 2 + len(unit) > chunk_size:
            chunks.append("\n\n".join(current))
            tail = _overlap_tail(current, overlap)
            # Keep the size guarantee: if the overlap tail plus this unit wouldn't
            # fit, drop the overlap at this one boundary rather than exceed the cap.
            if tail and _joined_len(tail) + 2 + len(unit) > chunk_size:
                tail = []
            current = tail
        current.append(unit)

    if current:
        chunks.append("\n\n".join(current))
    return chunks
