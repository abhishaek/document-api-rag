"""Tests for the boilerplate cleaner (app/services/cleaning_service.py).

The cleaner's contract is asymmetric: it must remove high-confidence page
furniture (TOC leaders, running headers/footers, page numbers) while never
touching real prose. These tests pin both halves — what goes, and what stays.
"""

from app.services.cleaning_service import clean_markdown


def test_table_of_contents_leaders_are_removed():
    """Dotted-leader lines are a table of contents — pure noise to a retriever.
    Embedded in a real document (the realistic case), only the TOC lines go."""
    markdown = (
        "AWS Fargate ...................................................... 39\n"
        "AWS Lambda ....................................................... 39\n"
        "AWS Serverless Application Repository ............................ 40\n"
        "AWS Lambda lets you run code without provisioning servers."
    )

    cleaned = clean_markdown(markdown)

    assert "....." not in cleaned
    assert "AWS Lambda lets you run code without provisioning servers." in cleaned


def test_running_header_repeated_across_pages_is_removed():
    """A short line recurring on every page is a running header, not content."""
    header = "Overview of Amazon Web Services AWS Whitepaper"
    markdown = "\n".join(
        [
            header,
            "Lambda lets you run code without managing servers.",
            header,
            "You pay only for the compute time you consume.",
            header,
            "Just upload your code and Lambda runs it.",
            header,
            "It scales automatically with high availability.",
        ]
    )

    cleaned = clean_markdown(markdown)

    assert header not in cleaned
    # The real sentences survive intact.
    assert "Lambda lets you run code without managing servers." in cleaned
    assert "It scales automatically with high availability." in cleaned


def test_bare_page_numbers_are_removed():
    markdown = "Some real content about Lambda.\n39\nMore real content."

    cleaned = clean_markdown(markdown)

    assert "39" not in cleaned.split("\n")
    assert "Some real content about Lambda." in cleaned
    assert "More real content." in cleaned


def test_prose_is_left_untouched():
    """Nothing in ordinary prose trips a rule: no 4+ dots, no verbatim repeats,
    no bare-number lines."""
    markdown = (
        "AWS Lambda lets you run code without provisioning servers.\n"
        "You pay only for the compute time you consume.\n\n"
        "It scales automatically, so you focus on the code, not the servers."
    )

    assert clean_markdown(markdown) == markdown


def test_ellipsis_in_prose_is_not_treated_as_a_leader():
    """Three dots (an ellipsis) is real punctuation; only 4+ signal a TOC."""
    markdown = "Well... Lambda runs your code and scales it for you."

    assert clean_markdown(markdown) == markdown


def test_blank_lines_are_preserved_for_block_structure():
    """The chunker splits on blank lines, so cleaning must not collapse them."""
    markdown = "First paragraph about Lambda.\n\nSecond paragraph about scaling."

    assert clean_markdown(markdown) == markdown


def test_all_boilerplate_falls_back_to_original():
    """If every line looks like furniture, return the input rather than nothing —
    over-firing rules must not silently empty a document."""
    markdown = "Heading ....... 1\n2\nHeading ....... 1\n2"

    # Every line is a leader or a page number; cleaning would empty it, so the
    # original is returned instead of an empty string.
    assert clean_markdown(markdown) == markdown


def test_repeated_line_below_threshold_survives():
    """A line repeated only a couple of times is not confidently a header — a
    real sentence can recur — so it is kept."""
    line = "Lambda scales automatically."
    markdown = "\n".join([line, "Other content here.", line])

    assert line in clean_markdown(markdown)
