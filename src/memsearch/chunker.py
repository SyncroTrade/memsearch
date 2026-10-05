"""Markdown chunking — split markdown files into semantic chunks by headings."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Minimum meaningful text length after stripping metadata noise.
# Chunks with less useful text than this are dropped during chunking.
_MIN_MEANINGFUL_LEN = 2

# Hard cap on stored heading length. Keeps headings well under the
# MilvusStore VARCHAR(1024) schema limit for the `heading` field
# (see store.py) with headroom for future schema changes, and headings
# past this length have no retrieval value anyway.
_MAX_HEADING_BYTES = 512

CHUNK_MODES = ("section", "block")

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# A fence opener: three or more backticks (no further backtick on the line,
# which would make it inline code) or three or more tildes.
_FENCE_OPEN_RE = re.compile(r"^\s*(?:(`{3,})[^`]*|(~{3,}).*)$")


def _clamp_heading(text: str, max_bytes: int = _MAX_HEADING_BYTES) -> str:
    """Truncate *text* to at most *max_bytes* UTF-8 bytes, without splitting
    a multi-byte character.

    A markdown heading line has no length limit, but the chunk store does
    (VARCHAR(1024) on `heading` — see store.py). An over-length heading with
    body text previously caused the whole file's chunks to be rejected by
    the store *after* its stale chunks had already been deleted, silently
    dropping the file from the index. Clamping at chunk-construction time —
    once, at the source of the heading string — keeps every downstream
    ``Chunk(heading=...)`` construction safe without touching the store
    schema.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # errors="ignore" drops any partial multi-byte sequence left dangling
    # at the truncation boundary.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def clean_content_for_embedding(text: str) -> str:
    """Strip metadata noise from chunk content before embedding.

    Removes HTML comments (<!-- ... -->), which often contain session/turn
    UUIDs and transcript paths that dilute embedding quality.  The original
    content stored in Milvus is unchanged — this only affects the text
    sent to the embedding model.
    """
    cleaned = _HTML_COMMENT_RE.sub("", text)
    # Collapse runs of blank lines left behind by removed comments
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _has_meaningful_content(text: str) -> bool:
    """Return True if *text* has enough substance to be worth indexing.

    Strips HTML comments, heading lines, and whitespace, then checks
    whether the remaining body text meets the minimum length threshold.
    A section like ``## Session 03:16`` with no body is rejected, while
    ``## Title\\nSome real content`` is kept.
    """
    # Remove HTML comments first
    stripped = _HTML_COMMENT_RE.sub("", text)
    # Remove heading lines — we only care about body text
    lines = [ln for ln in stripped.splitlines() if not _HEADING_RE.match(ln)]
    body = "\n".join(lines).strip()
    return len(body) >= _MIN_MEANINGFUL_LEN


@dataclass(frozen=True)
class Chunk:
    """A single chunk extracted from a markdown document."""

    content: str
    source: str  # file path
    heading: str  # nearest heading (empty string for preamble)
    heading_level: int  # 0 for preamble
    start_line: int
    end_line: int
    content_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", compute_content_hash(self.content))


def compute_content_hash(content: str) -> str:
    """Compute the content hash embedded in a chunk's composite id.

    A pure function of *content* alone -- it does not depend on source,
    line range, heading, or model. That purity is what makes re-keying
    possible: a chunk whose text is untouched keeps the same
    content_hash even after an earlier insertion shifts its line range
    (see #704). Kept in sync with ``Chunk.__post_init__`` below so
    callers that only have a stored ``content`` string (no ``Chunk``
    instance) can recompute the same hash.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_chunk_id(
    source: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    model: str,
) -> str:
    """Compute a composite chunk ID matching OpenClaw's format.

    ``hash(source:path:startLine:endLine:contentHash:model)``
    """
    raw = f"markdown:{source}:{start_line}:{end_line}:{content_hash}:{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_markdown(
    text: str,
    source: str = "",
    *,
    max_chunk_size: int = 1500,
    overlap_lines: int = 2,
    mode: str = "section",
) -> list[Chunk]:
    """Split markdown *text* into chunks, breaking on headings.

    Chunks that exceed *max_chunk_size* characters are split further at
    paragraph boundaries.  A small *overlap_lines* context is carried
    forward to preserve continuity.

    *mode* ``"section"`` (the default) emits one chunk per heading section.
    ``"block"`` emits one chunk per list item, table or paragraph inside each
    section; see :func:`_chunk_blocks`.  Any other value raises ``ValueError``.

    Block mode strips the ``\\r`` of CRLF lines before it looks for headings, so
    CRLF text chunks like LF text. Section mode keeps the ``\\r``, which means a
    heading whose title is empty apart from ``\\r`` (``"### \\r"``), or text with
    a lone ``\\r`` line ending, can be sectioned differently by the two modes.
    """
    if mode not in CHUNK_MODES:
        raise ValueError(f"unknown chunk mode {mode!r}; expected one of {', '.join(CHUNK_MODES)}")
    lines = text.split("\n")
    if mode == "block":
        # Section mode keeps the "\r" of a CRLF line inside chunk content;
        # block mode drops it so CRLF text chunks exactly like LF text.
        lines = [ln.removesuffix("\r") for ln in lines]
    # Find all heading positions
    heading_positions: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            heading_positions.append((i, len(m.group(1)), _clamp_heading(m.group(2).strip())))

    # Build sections between headings
    sections: list[tuple[int, int, str, int]] = []  # (start, end, heading, level)
    if not heading_positions or heading_positions[0][0] > 0:
        end = heading_positions[0][0] if heading_positions else len(lines)
        sections.append((0, end, "", 0))

    for idx, (line_idx, level, title) in enumerate(heading_positions):
        next_start = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        sections.append((line_idx, next_start, title, level))

    if mode == "block":
        return _chunk_blocks(lines, sections, source=source, max_size=max_chunk_size)

    chunks: list[Chunk] = []
    for start, end, heading, level in sections:
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text or not _has_meaningful_content(section_text):
            continue

        if len(section_text) <= max_chunk_size:
            chunks.append(
                Chunk(
                    content=section_text,
                    source=source,
                    heading=heading,
                    heading_level=level,
                    start_line=start + 1,
                    end_line=end,
                )
            )
        else:
            # Split large sections at paragraph boundaries
            chunks.extend(
                _split_large_section(
                    lines[start:end],
                    source=source,
                    heading=heading,
                    heading_level=level,
                    base_line=start,
                    max_size=max_chunk_size,
                    overlap=overlap_lines,
                )
            )

    return chunks


def _fence_marker(line: str) -> str:
    """Return the fence run (``` or ~~~, any length >= 3) opening on *line*, or ""."""
    m = _FENCE_OPEN_RE.match(line)
    return (m.group(1) or m.group(2)) if m else ""


def _fence_end(lines: list[str], i: int, end: int, marker: str) -> int:
    """Return the index just past the fence opened on line *i* (or *end* if unclosed)."""
    for j in range(i + 1, end):
        s = lines[j].strip()
        if len(s) >= len(marker) and s == marker[0] * len(s):
            return j + 1
    return end


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _parse_blocks(lines: list[str], start: int, end: int) -> list[tuple[int, int, str]]:
    """Group ``lines[start:end]`` into blocks: ``(first_idx, last_idx, kind)``.

    Blank lines separate blocks and belong to none (except inside a fence).
    Kinds: ``"list"`` (an item plus its continuation lines), ``"table"``
    (consecutive lines starting with ``|``) and ``"paragraph"``.  A fenced
    run is consumed whole wherever it starts, blank lines and list-looking
    lines inside it included; an unclosed fence runs to *end*.
    """
    blocks: list[tuple[int, int, str]] = []
    i = start
    while i < end:
        if not lines[i].strip():
            i += 1
            continue
        if _is_table_row(lines[i]):
            j = i + 1
            while j < end and _is_table_row(lines[j]):
                j += 1
            blocks.append((i, j - 1, "table"))
            i = j
            continue
        kind = "list" if _LIST_ITEM_RE.match(lines[i]) else "paragraph"
        j = i
        while j < end:
            marker = _fence_marker(lines[j])
            j = _fence_end(lines, j, end, marker) if marker else j + 1
            if j >= end or not lines[j].strip() or _is_table_row(lines[j]) or _LIST_ITEM_RE.match(lines[j]):
                break
        last = j - 1
        while not lines[last].strip():  # an unclosed fence can run into trailing blanks
            last -= 1
        blocks.append((i, last, kind))
        i = j
    return blocks


def _merge_empty_blocks(lines: list[str], blocks: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge each block with no meaningful content into its neighbour (see ``_chunk_blocks``).

    Returns the blocks with a one-line paragraph retyped ``"label"``.
    """
    merged: list[tuple[int, int, str]] = []
    pending: int | None = None  # first line of empty blocks waiting for the next block
    for first, last, kind in blocks:
        if kind == "paragraph" and first == last:
            kind = "label"  # a one-line paragraph stays a label candidate whatever is merged into it
        if not _has_meaningful_content("\n".join(lines[first : last + 1])):
            if merged:
                merged[-1] = (merged[-1][0], last, merged[-1][2])
            elif pending is None:
                pending = first
            continue
        if pending is not None:
            first, pending = pending, None
        merged.append((first, last, kind))
    if pending is not None:  # no block has content of its own
        merged.append((pending, blocks[-1][1], "paragraph"))
    return merged


def _merge_empty_pieces(
    lines: list[str], pieces: list[tuple[str, int, int]], max_size: int
) -> list[tuple[str, int, int]]:
    """After the cap has cut blocks into pieces, merge a piece with no meaningful
    content (a comment cut loose from its text) into the previous piece, or the
    next one, whenever the merged text still fits *max_size*."""

    def _join(a: tuple[str, int, int], b: tuple[str, int, int]) -> tuple[str, int, int]:
        content = "\n".join(lines[a[1] : b[2] + 1]).strip()
        return content, a[1], b[2]

    todo = list(pieces)
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(todo):
        piece = todo[i]
        if not _has_meaningful_content(piece[0]):
            if out and len(_join(out[-1], piece)[0]) <= max_size:
                out[-1] = _join(out[-1], piece)
                i += 1
                continue
            if i + 1 < len(todo) and len(_join(piece, todo[i + 1])[0]) <= max_size:
                todo[i + 1] = _join(piece, todo[i + 1])
                i += 1
                continue
        out.append(piece)
        i += 1
    return out


def _piece_chunks(lines: list[str], first: int, last: int, max_size: int) -> list[tuple[str, int, int]]:
    """Cut ``lines[first:last + 1]`` into ``(content, first_idx, last_idx)`` pieces.

    The gate is section mode's own: a text of at most *max_size* characters
    (after stripping) stays whole; a longer one is packed greedily into
    pieces of whole lines, each at most *max_size* characters, no overlap.
    A line of *max_size* characters or more is cut by ``_split_long_text``,
    the routine section mode uses for such a line.
    """

    def _span(idxs: list[int]) -> tuple[str, int, int] | None:
        content = "\n".join(lines[k] for k in idxs).strip()
        if not content:
            return None
        non_blank = [k for k in idxs if lines[k].strip()]
        return content, non_blank[0], non_blank[-1]

    whole = _span(list(range(first, last + 1)))
    if whole is None:
        return []
    if len(whole[0]) <= max_size:
        return [whole]

    pieces: list[tuple[str, int, int]] = []
    current: list[int] = []

    def _flush() -> None:
        span = _span(current) if current else None
        if span is not None:
            pieces.append(span)
        current.clear()

    for i in range(first, last + 1):
        line = lines[i]
        if len(line) >= max_size:
            _flush()
            pieces.extend((part.strip(), i, i) for part in _split_long_text(line, max_size) if part.strip())
        elif current and len("\n".join(lines[k] for k in [*current, i]).strip()) > max_size:
            _flush()
            current.append(i)
        else:
            current.append(i)
    _flush()
    return pieces


def _chunk_blocks(
    lines: list[str],
    sections: list[tuple[int, int, str, int]],
    *,
    source: str,
    max_size: int,
) -> list[Chunk]:
    """Emit one chunk per block inside each section (``mode="block"``).

    Sections are the ones section mode finds, with the same skips. Inside a
    section the steps run in this order:

    1. A block with no meaningful content of its own (only HTML comments, or
       under two characters) is merged into the previous block of the section,
       or into the next one when it is the first, so no line is lost and no
       chunk is left with nothing to embed. If no block has any, the whole
       body is one block.
    2. A one-line paragraph followed by a list item or a table is joined to it
       (a label).
    3. The heading line joins the first block.

    A block is cut into pieces when it exceeds the cap; a piece left with no
    meaningful content (a comment cut loose from its text, or a heading with
    only a comment) is merged into a neighbouring piece of the section when
    the merged text still fits the cap, and otherwise stays as it is.

    No line appears in two chunks; ``overlap_lines`` plays no part.

    Sections are found exactly as section mode finds them, so a heading-looking
    line inside a fence still starts a new section and cuts the fence. A fence
    must start its line (after indentation): one opened on a list item's own
    line (``- ```) is not recognised.
    """
    chunks: list[Chunk] = []
    for start, end, heading, level in sections:
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text or not _has_meaningful_content(section_text):
            continue

        blocks = _merge_empty_blocks(lines, _parse_blocks(lines, start + 1 if level else start, end))
        spans: list[tuple[int, int]] = []
        k = 0
        while k < len(blocks):
            first, last, kind = blocks[k]
            if kind == "label" and k + 1 < len(blocks) and blocks[k + 1][2] in ("list", "table"):
                last = blocks[k + 1][1]
                k += 1
            spans.append((first, last))
            k += 1
        if level and spans:
            spans[0] = (start, spans[0][1])

        pieces = [piece for first, last in spans for piece in _piece_chunks(lines, first, last, max_size)]
        for content, start_idx, end_idx in _merge_empty_pieces(lines, pieces, max_size):
            chunks.append(
                Chunk(
                    content=content,
                    source=source,
                    heading=heading,
                    heading_level=level,
                    start_line=start_idx + 1,
                    end_line=end_idx + 1,
                )
            )
    return chunks


def _split_large_section(
    lines: list[str],
    *,
    source: str,
    heading: str,
    heading_level: int,
    base_line: int,
    max_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a large section into smaller chunks.

    Split priority: paragraph boundary > line boundary > sentence/char boundary.

    The last *overlap* lines of each emitted chunk are carried into the next
    one as context. Carried lines are never payload: when carry plus the next
    source line would reach *max_size*, carried lines are dropped from the
    front (all of them if need be) so a chunk is never pushed over the cap and
    cut mid-line. Every chunk is therefore a run of whole, consecutive source
    lines, unless a single source line is itself over *max_size*.
    """
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0
    # True while current_lines holds only lines carried over a paragraph split.
    carry_only = False

    def _emit(content: str, start_line: int, end_line: int) -> None:
        if content:
            chunks.append(
                Chunk(
                    content=content,
                    source=source,
                    heading=heading,
                    heading_level=heading_level,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    def _emit_bounded(content: str, start_line: int, end_line: int) -> None:
        content = content.strip()
        if not content:
            return
        if len(content) > max_size:
            for part in _split_long_text(content, max_size):
                _emit(part.strip(), start_line, end_line)
        else:
            _emit(content, start_line, end_line)

    for i, line in enumerate(lines):
        if carry_only:
            # The paragraph path carries lines before it knows the next one;
            # drop carry that would push this line's chunk to the cap.
            while current_lines and len("\n".join([*current_lines, line])) >= max_size:
                current_lines.pop(0)
                current_start += 1
            carry_only = False
        current_lines.append(line)
        text = "\n".join(current_lines)

        is_paragraph_break = line.strip() == "" and i + 1 < len(lines)
        is_last_line = i == len(lines) - 1

        # Preferred: split at paragraph boundary
        if len(text) >= max_size and is_paragraph_break:
            _emit_bounded(text, base_line + current_start + 1, base_line + i + 1)
            overlap_start = max(0, len(current_lines) - overlap)
            current_lines = current_lines[overlap_start:]
            current_start = i + 1 - len(current_lines)
            carry_only = True
            continue

        # Forced line-boundary split: no paragraph break found but text is
        # too large.  Roll back the current line so the previous lines form
        # a chunk and the current line starts the next one.
        if len(text) >= max_size and not is_paragraph_break and len(current_lines) > 1:
            current_lines.pop()
            content = "\n".join(current_lines).strip()
            _emit_bounded(content, base_line + current_start + 1, base_line + i)
            overlap_start = max(0, len(current_lines) - overlap)
            current_lines = current_lines[overlap_start:]
            # Overlap is context, never payload: shed carry that would
            # push the re-added line's chunk to the cap.
            while current_lines and len("\n".join([*current_lines, line])) >= max_size:
                current_lines.pop(0)
            current_lines.append(line)  # re-add the rolled-back line
            current_start = i - len(current_lines) + 1
            continue

        # Single line exceeds max_size — split within the line
        if len(text) >= max_size and len(current_lines) == 1:
            sub_chunks = _split_long_text(text, max_size)
            for part in sub_chunks:
                _emit(part.strip(), base_line + current_start + 1, base_line + i + 1)
            current_lines = []
            current_start = i + 1
            continue

        if is_last_line:
            _emit_bounded(text, base_line + current_start + 1, base_line + i + 1)
            current_lines = []

    # Flush any remaining content (e.g. a rolled-back line from the last
    # iteration that never got a chance to be emitted).
    if current_lines:
        remaining = "\n".join(current_lines).strip()
        if remaining:
            end_line = base_line + len(lines)
            start_line = base_line + current_start + 1
            _emit_bounded(remaining, start_line, end_line)

    return chunks


# Sentence-ending punctuation for splitting long text without line breaks.
# CJK punctuation (fullwidth stop/exclaim/question/semicolon + ellipsis)
# always acts as a boundary. ASCII punctuation (.!?;) only counts when
# followed by whitespace, end-of-string, or a CJK character -- so
# `user@example.com`, `path/to/file.py`, `http://foo.bar`, and `v1.2.3`
# are not split mid-token.
_SENTENCE_END_RE = re.compile(
    r"(?:……|…|[。\uFF01\uFF1F\uFF1B]\s*|[.!?;](?=\s|$|[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF])\s*)"
)


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split a long string that has no line breaks into pieces ≤ *max_size*.

    Prefers splitting at sentence boundaries; falls back to character position.
    """
    parts: list[str] = []
    while len(text) > max_size:
        # Look for the last sentence boundary within max_size
        best = -1
        for m in _SENTENCE_END_RE.finditer(text, 0, max_size):
            best = m.end()
        if best > 0:
            parts.append(text[:best])
            text = text[best:]
        else:
            # No sentence boundary found — hard split at max_size
            parts.append(text[:max_size])
            text = text[max_size:]
    if text:
        parts.append(text)
    return parts
