"""Strip document boilerplate from parsed Markdown before chunking.

PDF extraction carries artifacts that are pure noise to a retriever: a table of
contents (dozens of dotted-leader lines), running page headers/footers repeated
on every page, and stray page-number lines. Embedded as chunks they waste recall
slots and match on surface tokens — a TOC line like ``AWS Lambda .......... 39``
scores against a "what is AWS Lambda" query while carrying no actual content, and
a running header bloats every chunk it lands in.

Cleaning is deliberately **conservative**: it removes only high-confidence
boilerplate, because deleting real prose hurts retrieval far more than leaving a
little noise. Three signals, all source-agnostic (no per-document hardcoding):

* **Dotted leaders** — a run of four or more dots only ever appears in a table of
  contents or index, never in prose.
* **Repeated short lines** — a short line that recurs many times across the
  document is a running header/footer, not content. Prose sentences essentially
  never repeat verbatim that often; page furniture repeats once per page.
* **Bare page-number lines** — a line that is nothing but a small integer.

Pure and synchronous, like the chunker it feeds: no I/O, trivially unit-tested.
"""

import re
from collections import Counter

# Four or more dots: a table-of-contents / index leader ("Topic ...... 39"). Three
# dots (an ellipsis) is left alone, so ordinary prose is never touched.
_DOT_LEADER = re.compile(r"\.{4,}")

# A line counts as a running header/footer when it recurs at least this many times
# and is no longer than this. Both bounds guard against removing real content: a
# genuine sentence rarely repeats verbatim, and headers/footers are short.
_MIN_REPEATS = 4
_MAX_BOILERPLATE_LEN = 80


def _is_page_number(line: str) -> bool:
    """True for a line that is only a small integer — an isolated page number."""
    stripped = line.strip()
    return stripped.isdigit() and len(stripped) <= 4


def clean_markdown(markdown: str) -> str:
    """Remove boilerplate lines (TOC leaders, running headers/footers, page
    numbers) from parsed Markdown, preserving everything else verbatim.

    Blank lines are kept, so the block structure the chunker relies on is intact.
    Returns the input unchanged if cleaning would leave nothing but whitespace —
    a signal the rules over-fired (or the document really was all furniture), in
    which case the caller is better served by the raw text than by nothing.
    """
    lines = markdown.split("\n")

    # Count short, non-blank lines to find the ones that recur like page furniture.
    counts = Counter(
        stripped
        for line in lines
        if (stripped := line.strip()) and len(stripped) <= _MAX_BOILERPLATE_LEN
    )
    repeated = {line for line, n in counts.items() if n >= _MIN_REPEATS}

    kept: list[str] = []
    for line in lines:
        if _DOT_LEADER.search(line):
            continue  # table-of-contents / index leader
        if _is_page_number(line):
            continue  # isolated page number
        if line.strip() in repeated:
            continue  # running header/footer
        kept.append(line)

    cleaned = "\n".join(kept)
    return cleaned if cleaned.strip() else markdown
