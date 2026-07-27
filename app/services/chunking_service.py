"""Split parsed Markdown into overlapping chunks for embedding.

Chunking is the biggest lever on retrieval quality after the embedding model:
too large and a chunk dilutes its own meaning (the embedding averages several
topics, so nothing matches sharply); too small and it loses the context that
makes it answerable. The target here is ~1200 characters per chunk with ~150
characters of overlap, so a passage split across a boundary still appears whole
in one of the two neighbouring chunks.

The algorithm splits on the **strongest boundary that helps**, dropping to a
finer one only for a piece that is still larger than a chunk:

    paragraph (blank line)  ->  sentence  ->  line  ->  character window

This ordering is what keeps a chunk topically pure. Prose extracted from a PDF
often arrives with almost no blank lines — one giant run of text — so the
paragraph level alone can't help and the old approach fell straight to blind
character windows, cutting mid-sentence and averaging two adjacent sections
(say, "AWS Fargate" and "AWS Lambda") into one muddy vector. Breaking on
*sentence* boundaries instead means one section's sentences fill its chunks and
the next section starts fresh, so a query for one topic stops matching chunks
that are mostly about its neighbour.

A run with no usable boundary at all (a long table row, unbroken code, a
spaceless token) falls back to a fixed character window — still overlapping — as
a last resort.

Once the text is broken into boundary-respecting *atoms*, whole atoms are packed
greedily into chunks up to ``chunk_size``, and each new chunk is seeded with the
trailing atom(s) of the previous one (up to the overlap budget) so continuity is
preserved without cutting text.

Pure and synchronous: no I/O, so it's trivial to unit-test and cheap to call
inline from the ingestion pipeline.
"""

import re

# Library defaults for standalone/test use. The application overrides these per
# request from config (settings.chunk_size / chunk_overlap -> passed by
# ingestion_service), so tuning in production is a .env change, not a code edit.
# Targets, not hard physics: a chunk never *exceeds* chunk_size, but may be
# shorter when a boundary falls early.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150

# One or more blank lines (blank = nothing but whitespace) separate Markdown
# blocks. A single newline inside a block (a soft wrap) is left intact.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# A sentence ends at '.', '!' or '?' followed by whitespace. Deliberately
# approximate: it over-splits on abbreviations ("U.S. ", "e.g. ") and decimals
# written oddly, but a chunk that begins one sentence early is far cheaper than
# one that averages two topics into a blurry vector. The whitespace after the
# terminator is consumed by the split, so a period at a PDF line wrap
# ("servers.\nYou pay...") still breaks cleanly.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    """Break Markdown into stripped, non-empty blocks on blank lines."""
    return [block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip()]


def _joined_len(atoms: list[str]) -> int:
    """Length of ``"\\n\\n".join(atoms)`` without building the string."""
    if not atoms:
        return 0
    return sum(len(a) for a in atoms) + 2 * (len(atoms) - 1)


def _char_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an unbreakable run into overlapping fixed-size character windows.

    The last resort, for a run with no paragraph, sentence, or line boundary to
    break on (a long table row, unbroken code): the window slides by
    ``chunk_size - overlap`` so each piece shares ``overlap`` characters with the
    next. Unlike the higher levels this cuts mid-word — but only once every
    meaningful boundary has been exhausted.
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


# Ordered strongest -> weakest. Each divides text into pieces; a level is only
# tried when the level above left a piece still larger than a chunk. Character
# windows (below these) are the terminal fallback.
_BOUNDARIES = (
    _PARAGRAPH_BREAK.split,
    _SENTENCE_BREAK.split,
    lambda text: text.split("\n"),
)


def _atomize(text: str, chunk_size: int, overlap: int, level: int = 0) -> list[str]:
    """Break text into order-preserving atoms, each at most ``chunk_size`` chars.

    Splits on the strongest boundary in ``_BOUNDARIES`` that actually divides the
    text, recursing to a finer boundary only for a piece that is still too large.
    When every boundary has been exhausted (a run with none), the piece is cut
    into overlapping character windows.

    Atoms are returned in reading order and carry no overlap between them (the
    packer in ``chunk_markdown`` adds that); the character-window fallback is the
    one exception, since a boundary-less run has no whole unit to carry over.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if level >= len(_BOUNDARIES):
        return _char_windows(text, chunk_size, overlap)

    pieces = [p for p in _BOUNDARIES[level](text) if p.strip()]
    if len(pieces) <= 1:
        # This boundary didn't divide the text — drop to a finer one on the same
        # text rather than recursing on an identical single piece (which would
        # loop). Character windows wait at the bottom for text nothing splits.
        return _atomize(text, chunk_size, overlap, level + 1)

    atoms: list[str] = []
    for piece in pieces:
        atoms.extend(_atomize(piece, chunk_size, overlap, level + 1))
    return atoms


def _overlap_tail(atoms: list[str], overlap: int) -> list[str]:
    """The longest suffix of ``atoms`` whose joined length is <= ``overlap``.

    Seeds the next chunk with as much trailing context as the overlap budget
    allows, kept to whole atoms. May be empty if even the last atom alone exceeds
    the budget (then that boundary carries no atom overlap — a character-window
    run, already self-overlapping, is the usual reason).
    """
    tail: list[str] = []
    for atom in reversed(atoms):
        candidate_len = len(atom) + (2 if tail else 0) + _joined_len(tail)
        if candidate_len > overlap:
            break
        tail.insert(0, atom)
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

    # Normalise blank-line runs and edge whitespace up front so blocks are joined
    # by exactly one blank line, then break everything into boundary-respecting
    # atoms the packer below can treat uniformly.
    normalized = "\n\n".join(_split_paragraphs(markdown))
    atoms = _atomize(normalized, chunk_size, overlap)

    chunks: list[str] = []
    current: list[str] = []
    for atom in atoms:
        # Would adding this atom overflow the current chunk? Emit and reset first.
        if current and _joined_len(current) + 2 + len(atom) > chunk_size:
            chunks.append("\n\n".join(current))
            tail = _overlap_tail(current, overlap)
            # Keep the size guarantee: if the overlap tail plus this atom wouldn't
            # fit, drop the overlap at this one boundary rather than exceed the cap.
            if tail and _joined_len(tail) + 2 + len(atom) > chunk_size:
                tail = []
            current = tail
        current.append(atom)

    if current:
        chunks.append("\n\n".join(current))
    return chunks
