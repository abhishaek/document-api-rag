"""Tests for the document parser (app/services/parsing_service.py).

Each supported prose type is parsed to Markdown and checked for its known text,
so a converter silently dropping content is caught. Binary fixtures (pdf, docx)
are built in-memory rather than committed: no writer library is installed, and a
generated fixture keeps the expected text and the assertion in one place.

The failure paths matter as much as the happy ones — a corrupt or empty document
must raise a domain error the pipeline can turn into a `failed` status, and a
type with no parser (the deferred tabular lane) must fail loudly, not silently.
"""

import io
import zipfile

import pytest

from app.services.parsing_service import (
    NoParserRegisteredError,
    UnparseableDocumentError,
    parse_to_markdown,
)

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _make_docx(text: str) -> bytes:
    """A minimal valid .docx (a zip of OOXML parts) carrying one paragraph."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'  # noqa: E501
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'  # noqa: E501
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'  # noqa: E501
            "</Relationships>",
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


def _make_pdf(text: str) -> bytes:
    """A minimal single-page .pdf with one line of extractable text."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode() + b") Tj ET"
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objs) + 1,
        xref_pos,
    )
    return out


async def test_markdown_passes_through_intact():
    raw = b"# Title\n\nBody **text** with emphasis.\n"

    result = await parse_to_markdown(raw, "text/markdown", "doc.md")

    assert "# Title" in result
    assert "**text**" in result


async def test_plain_text_is_returned():
    raw = b"just some plain text content\n"

    result = await parse_to_markdown(raw, "text/plain", "doc.txt")

    assert "just some plain text content" in result


async def test_html_is_converted_to_markdown():
    raw = b"<h1>Heading</h1><p>A paragraph.</p>"

    result = await parse_to_markdown(raw, "text/html", "doc.html")

    # The <h1> became a Markdown heading rather than being passed through as tags.
    assert "# Heading" in result
    assert "A paragraph." in result
    assert "<h1>" not in result


async def test_docx_text_is_extracted():
    raw = _make_docx("Quarterly report body text")

    result = await parse_to_markdown(raw, DOCX_MIME, "report.docx")

    assert "Quarterly report body text" in result


async def test_pdf_text_is_extracted():
    raw = _make_pdf("Hello from the PDF fixture")

    result = await parse_to_markdown(raw, PDF_MIME, "doc.pdf")

    assert "Hello from the PDF fixture" in result


@pytest.mark.parametrize("raw", [b"", b"   \n\t  \n"])
async def test_empty_or_blank_content_raises_unparseable(raw):
    """A file that parses to nothing usable is a failure, not an empty success —
    otherwise a scanned/image-only PDF would sail through and embed no text."""
    with pytest.raises(UnparseableDocumentError):
        await parse_to_markdown(raw, "text/plain", "blank.txt")


async def test_backend_exception_is_wrapped(monkeypatch):
    """Whatever a parser backend raises on bad input (pdfminer, mammoth, ...) must
    surface as a domain error, with the original chained for diagnosis — callers
    branch on document state, never on library internals.

    markitdown is deliberately resilient (it falls back to plain-text extraction
    rather than raising), so a failing backend is injected here instead of relying
    on a crafted corrupt file that markitdown would happily read as text.
    """
    import app.services.parsing_service as parsing

    def boom(raw: bytes, *, extension: str) -> str:
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(parsing._markitdown, "to_markdown", boom)

    with pytest.raises(UnparseableDocumentError) as exc_info:
        await parse_to_markdown(b"anything", PDF_MIME, "doc.pdf")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_unparseable_error_carries_context():
    """The error names the file and type so the pipeline can record why a
    document failed."""
    with pytest.raises(UnparseableDocumentError) as exc_info:
        await parse_to_markdown(b"", "text/plain", "blank.txt")

    assert exc_info.value.filename == "blank.txt"
    assert exc_info.value.mime == "text/plain"


@pytest.mark.parametrize(
    "mime",
    [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ],
)
async def test_tabular_types_have_no_parser(mime):
    """csv/xlsx are the deferred tabular lane: no parser is registered, so routing
    one here is a wiring error that must fail loudly (it can't reach here anyway —
    upload 415s it first)."""
    with pytest.raises(NoParserRegisteredError):
        await parse_to_markdown(b"a,b,c\n1,2,3\n", mime, "data.csv")
