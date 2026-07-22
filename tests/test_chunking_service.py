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
