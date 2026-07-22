"""Parse uploaded documents into Markdown for the ingestion pipeline.

markitdown converts each supported format to Markdown, the single text
representation the chunker consumes. Only *prose* types are handled here
(pdf / html / docx / txt / md); tabular types (csv / xlsx) are deliberately
absent — they belong to the structured / text-to-SQL lane, not this pipeline,
and are rejected at upload before they ever reach parsing.

Parsers sit behind a MIME-keyed registry — the same seam ``Storage`` gives blob
backends. Today one ``MarkitdownParser`` serves every entry, but a heavier
backend (OCR, or a hosted service for scanned PDFs) can later be registered for
a single MIME without changing callers.
"""

import asyncio
import io
import logging
from typing import Protocol

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Base class for parsing failures."""


class UnparseableDocumentError(ParsingError):
    """A supported document that a parser could not turn into usable text.

    Corrupt, empty, password-protected, or (for a PDF) image-only with no text
    layer. Distinct from a routing bug: the type *is* supported here — the bytes
    just couldn't be read. Maps to a `failed` document status, not an HTTP error.
    """

    def __init__(self, mime: str, filename: str, reason: str) -> None:
        self.mime = mime
        self.filename = filename
        self.reason = reason
        super().__init__(f"Could not parse {filename!r} ({mime}): {reason}")


class NoParserRegisteredError(ParsingError):
    """No parser is registered for a MIME type.

    Should be unreachable in normal operation: the upload allow-list mirrors this
    registry, so anything reaching here has already been vetted. Getting here
    means the two drifted — a wiring bug that must fail loudly rather than
    silently skip a document.
    """

    def __init__(self, mime: str) -> None:
        self.mime = mime
        super().__init__(f"No parser registered for MIME type: {mime}")


class Parser(Protocol):
    """Turns a document's bytes into Markdown.

    Synchronous and CPU-bound; ``parse_to_markdown`` runs implementations off the
    event loop. ``extension`` is the canonical suffix for the *verified* MIME
    (".pdf", ".docx", ...) and routes the bytes to the right converter — it is
    derived from the sniffed type, never the client's filename, so a mislabelled
    upload cannot misroute here.
    """

    def to_markdown(self, raw: bytes, *, extension: str) -> str: ...


class MarkitdownParser:
    """``Parser`` backed by markitdown. One instance serves every prose type."""

    def __init__(self) -> None:
        # Constructing MarkItDown loads magika's model, so build it once and
        # reuse it across calls rather than per document.
        self._md = MarkItDown()

    def to_markdown(self, raw: bytes, *, extension: str) -> str:
        result = self._md.convert_stream(io.BytesIO(raw), file_extension=extension)
        return result.markdown


# The prose allow-list, paired with the canonical extension markitdown routes on.
# The keys are the single source of truth for "what this service can parse"; they
# mirror ALLOWED_MIME_TYPES in document_service (minus the tabular types, which
# 415 at upload). Add a type here only alongside a parser that handles it.
_PROSE_MIME_EXTENSIONS: dict[str, str] = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}

_markitdown = MarkitdownParser()

# MIME -> Parser. One backend today; swap a single entry to route, say,
# application/pdf through an OCR backend later without touching callers.
_PARSERS: dict[str, Parser] = {mime: _markitdown for mime in _PROSE_MIME_EXTENSIONS}


async def parse_to_markdown(raw: bytes, mime: str, filename: str) -> str:
    """Parse a document's bytes to Markdown.

    ``mime`` is the type sniffed at upload (authoritative); ``filename`` is used
    only in error messages. Raises ``NoParserRegisteredError`` if the type is not
    supported, and ``UnparseableDocumentError`` if a supported type cannot be read
    or yields no text.
    """
    parser = _PARSERS.get(mime)
    if parser is None:
        raise NoParserRegisteredError(mime)
    extension = _PROSE_MIME_EXTENSIONS[mime]

    try:
        markdown = await asyncio.to_thread(
            parser.to_markdown, raw, extension=extension
        )
    except Exception as exc:
        # markitdown and its backends (pdfminer, mammoth, ...) raise a wide,
        # largely undocumented set of exceptions on bad input. Normalise them all
        # to one domain error so callers branch on document state, not library
        # internals. The original is logged for diagnosis.
        logger.warning(
            "parse failed",
            extra={"document_filename": filename, "mime": mime},
            exc_info=exc,
        )
        raise UnparseableDocumentError(mime, filename, str(exc)) from exc

    if not markdown or not markdown.strip():
        # A parser that succeeds but produces nothing usable: an empty file, or a
        # scanned / image-only PDF with no text layer. Embedding empty text is
        # pointless, so treat it as a failure the pipeline can mark and surface.
        raise UnparseableDocumentError(mime, filename, "parser produced no text")

    logger.info(
        "document parsed",
        extra={
            "document_filename": filename,
            "mime": mime,
            "markdown_chars": len(markdown),
        },
    )
    return markdown
