"""Tests for the Markdown chunker (app/services/chunking_service.py).

Small chunk_size / overlap values are used so the packing and overlap are easy to
verify by hand. The properties that matter: chunks never exceed the size cap,
consecutive chunks share context (overlap), no content is dropped, and paragraph
boundaries are respected until a single paragraph is too big to honour them.
"""

import pytest

from app.services.chunking_service import chunk_markdown


def test_empty_input_yields_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \t\n") == []


def test_short_document_is_a_single_chunk():
    assert chunk_markdown("Just one short paragraph.") == ["Just one short paragraph."]


def test_blank_lines_and_whitespace_are_normalised():
    """Leading/trailing whitespace and runs of blank lines collapse; blocks are
    joined by exactly one blank line."""
    messy = "  \n\n# Heading\n\n\n\nBody paragraph.\n\n  "

    assert chunk_markdown(messy) == ["# Heading\n\nBody paragraph."]


def test_paragraphs_pack_with_overlap():
    paragraphs = [
        "para one text",
        "para two text",
        "para three text",
        "para four text",
    ]
    markdown = "\n\n".join(paragraphs)

    chunks = chunk_markdown(markdown, chunk_size=35, overlap=15)

    assert len(chunks) == 3
    # Every chunk stays within the cap.
    assert all(len(c) <= 35 for c in chunks)
    # Consecutive chunks overlap by a shared paragraph (context isn't lost at the
    # boundary): "para two text" ends chunk 0 and begins chunk 1, etc.
    assert "para two text" in chunks[0] and "para two text" in chunks[1]
    assert "para three text" in chunks[1] and "para three text" in chunks[2]
    # No paragraph is dropped.
    for paragraph in paragraphs:
        assert any(paragraph in c for c in chunks)


def test_oversized_paragraph_falls_back_to_char_windows():
    """A single paragraph larger than chunk_size can't break on a boundary, so it
    is split into overlapping fixed-size windows."""
    # 250 distinct chars (a..z cycling), no blank lines -> one paragraph.
    text = "".join(chr(ord("a") + i % 26) for i in range(250))

    chunks = chunk_markdown(text, chunk_size=100, overlap=20)

    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
    # The window slides by chunk_size - overlap, so each chunk's last `overlap`
    # chars are the next chunk's first `overlap` chars.
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[1][-20:] == chunks[2][:20]


def test_chunks_never_exceed_the_size_cap():
    """Property check over a larger, mixed input."""
    paragraphs = [f"Paragraph number {i} " + "word " * (i % 40) for i in range(60)]
    markdown = "\n\n".join(paragraphs)

    chunks = chunk_markdown(markdown, chunk_size=500, overlap=50)

    assert chunks  # produced something
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_markdown("some text", chunk_size=100, overlap=100)


def test_structureless_prose_breaks_on_sentences_not_mid_sentence():
    """PDF-extracted prose arrives as one run with no blank lines. It must break
    on sentence boundaries, so no chunk starts or ends mid-sentence — the failure
    that let one section's text bleed into a neighbouring section's chunk."""
    sentences = [
        "Fargate runs containers without servers.",
        "It removes cluster management entirely.",
        "Lambda runs code without provisioning servers.",
        "You pay only for the compute you consume.",
    ]
    # Single-space separated: one paragraph, no blank lines, no newlines — the
    # shape markitdown produces for a PDF section.
    markdown = " ".join(sentences)

    chunks = chunk_markdown(markdown, chunk_size=90, overlap=45)

    assert all(len(c) <= 90 for c in chunks)
    # Every sentence survives intact somewhere — none was cut in half.
    for sentence in sentences:
        assert any(sentence in c for c in chunks)
    # Each chunk is composed of whole sentences: it starts at a capital and ends
    # at a sentence terminator, never mid-word.
    for chunk in chunks:
        assert chunk[0].isupper()
        assert chunk.rstrip()[-1] in ".!?"


def test_topics_land_in_separate_chunks():
    """The retrieval bug in miniature: two adjacent topics must not both dominate
    one chunk. With sentence-aware splitting each topic's sentences pack together
    rather than averaging into a single blurry chunk."""
    fargate = "Fargate is a serverless compute engine for containers. " * 3
    lambda_ = "Lambda lets you run code without managing any servers. " * 3
    markdown = (fargate + lambda_).strip()

    chunks = chunk_markdown(markdown, chunk_size=180, overlap=30)

    # No chunk carries substantial content from both topics: at least one of the
    # two topic words is absent from every chunk (some overlap at the single
    # boundary between them is fine, but they must not be thoroughly mixed).
    mixed = [c for c in chunks if "Fargate" in c and "Lambda" in c]
    assert len(mixed) <= 1


def test_long_sentence_without_boundaries_falls_back_to_char_windows():
    """A single 'sentence' larger than chunk_size with no line breaks has no
    finer boundary to honour, so it still degrades to overlapping char windows —
    the last resort is preserved."""
    # 250 chars, no spaces, no newlines, no sentence terminator -> nothing splits.
    text = "".join(chr(ord("a") + i % 26) for i in range(250))

    chunks = chunk_markdown(text, chunk_size=100, overlap=20)

    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][-20:] == chunks[1][:20]


def test_lines_are_used_when_sentences_are_too_long():
    """Content with no sentence punctuation (a bulleted list, config lines) still
    breaks on line boundaries rather than blindly windowing characters."""
    lines = [f"- configuration item number {i}" for i in range(6)]
    # No blank lines and no sentence terminators: only '\n' can divide this.
    markdown = "\n".join(lines)

    chunks = chunk_markdown(markdown, chunk_size=70, overlap=20)

    assert all(len(c) <= 70 for c in chunks)
    for line in lines:
        assert any(line in c for c in chunks)
    # No line was split across a char-window boundary: each item appears whole.
    for chunk in chunks:
        for fragment in chunk.split("\n\n"):
            assert fragment.startswith("- configuration item number")
