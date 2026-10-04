"""Tests for the markdown chunker."""

import random
from itertools import pairwise

from memsearch.chunker import _clamp_heading, chunk_markdown, clean_content_for_embedding


def test_simple_heading_split():
    md = """\
# Title

Some intro text.

## Section A

Content A here.

## Section B

Content B here.
"""
    chunks = chunk_markdown(md, source="test.md")
    assert len(chunks) >= 3
    headings = [c.heading for c in chunks]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings


def test_preamble_without_heading():
    md = "Just some text without any heading.\n\nMore text."
    chunks = chunk_markdown(md, source="test.md")
    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert chunks[0].heading_level == 0


def test_empty_input():
    chunks = chunk_markdown("", source="test.md")
    assert chunks == []


def test_content_hash_is_deterministic():
    md = "# Hello\n\nWorld"
    c1 = chunk_markdown(md, source="a.md")
    c2 = chunk_markdown(md, source="b.md")
    # Same content -> same content_hash, even different source
    assert c1[0].content_hash == c2[0].content_hash


def test_large_section_splitting():
    # Create a section larger than max_chunk_size
    paragraphs = [f"Paragraph {i}. " + "x" * 200 for i in range(20)]
    md = "# Big Section\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_markdown(md, source="test.md", max_chunk_size=500)
    assert len(chunks) > 1
    for c in chunks:
        assert c.heading == "Big Section"


def test_large_section_flushes_rolled_back_tail_line() -> None:
    """The final rolled-back line should still be emitted after the loop."""
    md = "# Big Section\n\n" + "\n".join(
        [
            "alpha " * 18,
            "beta " * 18,
            "tail " * 18,
        ]
    )

    chunks = chunk_markdown(md, source="test.md", max_chunk_size=120, overlap_lines=0)

    assert len(chunks) >= 2
    assert any("tail" in c.content for c in chunks)
    assert chunks[-1].end_line >= chunks[-1].start_line


def test_large_section_splits_long_line_after_forced_line_boundary() -> None:
    """A long line carried forward by rollback should not emit an oversized chunk."""
    max_size = 1500
    md = "\n".join(["short", "x" * 9500, "tail"])

    chunks = chunk_markdown(md, source="long-line.md", max_chunk_size=max_size)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= max_size for chunk in chunks)
    assert any("tail" in chunk.content for chunk in chunks)


def test_source_and_lines():
    md = "# A\n\nline1\n\n# B\n\nline2"
    chunks = chunk_markdown(md, source="doc.md")
    assert all(c.source == "doc.md" for c in chunks)
    assert chunks[0].start_line >= 1


def test_empty_session_heading_filtered():
    """Headings with no meaningful body text should be dropped."""
    md = """\
## Session 03:16

## Session 03:17

### 03:18
<!-- session:abc-123 turn:def-456 transcript:/path/to/file.jsonl -->
- User asked about something important and Claude provided a detailed answer.
"""
    chunks = chunk_markdown(md, source="daily.md")
    # The two empty "## Session" sections should be filtered out
    assert all("Session 03:16" not in c.heading for c in chunks)
    assert all("Session 03:17" not in c.heading for c in chunks)
    # The section with real content should survive
    assert len(chunks) >= 1
    assert any("something important" in c.content for c in chunks)


def test_html_comment_only_chunk_filtered():
    """A chunk with only a heading and HTML comment should be dropped."""
    md = """\
### 08:00
<!-- session:aaa-bbb turn:ccc-ddd transcript:/tmp/foo.jsonl -->
"""
    chunks = chunk_markdown(md, source="daily.md")
    assert chunks == []


def test_clean_content_for_embedding():
    """HTML comments should be stripped for embedding but content preserved."""
    text = (
        "### 03:18\n"
        "<!-- session:abc-123 turn:def-456 transcript:/path/to/file.jsonl -->\n"
        "- User asked about chunking optimization."
    )
    cleaned = clean_content_for_embedding(text)
    assert "session:abc-123" not in cleaned
    assert "transcript:" not in cleaned
    assert "chunking optimization" in cleaned


def test_clean_content_collapses_blank_lines():
    """Removing comments should not leave excessive blank lines."""
    text = "Line one.\n\n<!-- comment -->\n\n\n\nLine two."
    cleaned = clean_content_for_embedding(text)
    assert "\n\n\n" not in cleaned
    assert "Line one." in cleaned
    assert "Line two." in cleaned


def test_clean_content_handles_adjacent_html_comments() -> None:
    """Back-to-back HTML comments should not leave comment holes behind."""
    text = "Header\n<!-- first comment -->\n<!-- second comment -->\n\nBody text"

    cleaned = clean_content_for_embedding(text)

    assert "first comment" not in cleaned
    assert "second comment" not in cleaned
    assert cleaned == "Header\n\nBody text"


# --- Sentence-boundary splitting for long text without line breaks ---
#
# The chunker falls back to sentence-level splitting when a paragraph is
# longer than `max_chunk_size`. The tests below pin down the expected
# behavior of `_SENTENCE_END_RE`:
#   - CJK punctuation (fullwidth stop/exclaim/question/semicolon + ellipsis)
#     always counts as a boundary.
#   - ASCII punctuation (.!?;) only counts when followed by whitespace,
#     end-of-string, or a CJK character -- so engineering text like emails,
#     URLs, file extensions, and version numbers is not split mid-token.


def test_long_cjk_text_splits_on_cjk_sentence_boundaries() -> None:
    """Long Chinese text should split at CJK full-stop."""
    sentence = "这是一个用于测试中文分句行为的长句子。"
    text = sentence * 8

    chunks = chunk_markdown(text, source="zh.md", max_chunk_size=40)

    assert len(chunks) > 1
    assert all(chunk.content.endswith("。") for chunk in chunks[:-1])
    assert all(len(chunk.content) <= 40 for chunk in chunks)


def test_long_mixed_text_prefers_sentence_boundaries() -> None:
    """Mixed Chinese/English text still splits on sentence punctuation."""
    sentence = "请检查 Redis cache 是否命中。Then verify the fallback path works!"
    text = sentence * 5

    chunks = chunk_markdown(text, source="mixed.md", max_chunk_size=45)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 45 for chunk in chunks)
    # Each non-final chunk should end at a real sentence boundary.
    assert all(chunk.content.endswith(("。", "!")) for chunk in chunks[:-1])


def test_ascii_dot_in_engineering_text_is_not_a_boundary() -> None:
    """Dots inside emails, URLs, file paths, and version numbers should not split."""
    # Each token has ASCII dots immediately followed by non-whitespace, so the
    # lookahead in _SENTENCE_END_RE should reject them as boundaries. The
    # chunker is then forced to hard-split at max_chunk_size.
    tokens = [
        "user@example.com",
        "https://foo.bar/baz",
        "path/to/file.py",
        "memsearch.config.toml",
        "v1.2.3",
    ]
    text = " ".join(tokens * 8)

    chunks = chunk_markdown(text, source="engineering.md", max_chunk_size=60)

    assert len(chunks) > 1
    # No chunk boundary should split any of these tokens in half.
    joined = " || ".join(chunk.content for chunk in chunks)
    for token in tokens:
        assert token in joined, f"token {token!r} was broken across chunks"


def test_ascii_dot_followed_by_space_is_a_boundary() -> None:
    """Regular English sentences still split normally on `. ` / `! ` / `? `."""
    sentence = "This is a complete sentence. And here is another one! Is this a question? "
    text = sentence * 5

    chunks = chunk_markdown(text, source="en.md", max_chunk_size=60)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 60 for chunk in chunks)
    # Every non-final chunk should end at a sentence-ending punctuation.
    assert all(chunk.content.rstrip().endswith((".", "!", "?")) for chunk in chunks[:-1])


def test_long_cjk_text_splits_on_ellipsis() -> None:
    """Chinese ellipsis (……) acts as a sentence boundary when splitting."""
    sentence = "这个排查过程还没结束……但是系统已经记录了关键上下文……"
    text = sentence * 4

    chunks = chunk_markdown(text, source="ellipsis.md", max_chunk_size=35)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 35 for chunk in chunks)
    assert all(chunk.content.endswith("……") for chunk in chunks[:-1])


def test_long_cjk_text_splits_on_fullwidth_semicolon() -> None:
    """Chinese fullwidth semicolon (U+FF1B) acts as a sentence boundary."""
    semicolon = "\uff1b"
    sentence = f"先检查缓存命中率{semicolon}再确认索引是否完成{semicolon}"
    text = sentence * 5

    chunks = chunk_markdown(text, source="semicolon.md", max_chunk_size=24)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 24 for chunk in chunks)
    assert all(chunk.content.endswith(semicolon) for chunk in chunks[:-1])


def test_clamp_heading_noop_under_limit() -> None:
    assert _clamp_heading("Short heading") == "Short heading"


def test_clamp_heading_truncates_over_limit() -> None:
    heading = "x" * 2000
    clamped = _clamp_heading(heading)
    assert len(clamped.encode("utf-8")) <= 512
    assert clamped == "x" * 512


def test_clamp_heading_is_utf8_byte_safe() -> None:
    """A multi-byte character straddling the truncation boundary must not
    be split into an invalid partial sequence."""
    # U+4E2D ("中") is 3 bytes in UTF-8; repeating it means every truncation
    # boundary at a non-multiple-of-3 byte offset would land mid-character
    # if truncation were naive.
    heading = "中" * 1000
    clamped = _clamp_heading(heading, max_bytes=511)  # not a multiple of 3
    encoded = clamped.encode("utf-8")
    assert len(encoded) <= 511
    # Round-trips cleanly — no stray replacement/partial bytes.
    assert encoded.decode("utf-8") == clamped


def test_over_length_heading_with_body_is_clamped_not_dropped() -> None:
    """Regression for the VARCHAR(1024) heading rejection: a single-line
    heading far past the store's max_length, followed by real body text,
    must still produce chunks with a clamped (not rejected) heading, and
    the body text must not be lost."""
    heading_text = "A" * 1800
    md = f"# {heading_text}\n\nReal body content that makes this chunk meaningful.\n"
    chunks = chunk_markdown(md, source="test.md")
    assert len(chunks) >= 1
    assert all(len(c.heading.encode("utf-8")) <= 512 for c in chunks)
    assert any(c.heading == "A" * 512 for c in chunks)
    assert any("Real body content" in c.content for c in chunks)


def _line(tag: str, length: int) -> str:
    """A synthetic source line of exactly *length* characters (at least 8),
    unique per *tag*, with no leading or trailing whitespace."""
    return (tag + " ").ljust(length - 1, "x") + "."


def _assert_whole_source_lines(md: str, chunks, max_size: int = 1500) -> None:
    """Every chunk is a run of whole, consecutive source lines within the cap."""
    src = md.split("\n")
    for c in chunks:
        assert len(c.content) <= max_size
        # The stored range reproduces the content exactly.
        assert "\n".join(src[c.start_line - 1 : c.end_line]).strip() == c.content
        body = c.content.split("\n")
        # First and last line are whole source lines, not fragments of one.
        assert body[0] in src
        assert body[-1] in src
    # Nothing is lost: every body line sits in some chunk (a heading with no
    # body of its own is dropped by design).
    covered = {ln for c in chunks for ln in c.content.split("\n")}
    assert all(ln in covered for ln in src if ln.strip() and not ln.startswith("#"))


def test_overlap_carry_never_pushes_chunk_past_cap() -> None:
    """Carried overlap lines used to be re-added in front of the next line
    without checking the sum, so the chunk went over the cap and was cut in
    the middle of a source line. Carry must shrink instead."""
    lines = [_line("a", 434), _line("b", 729), _line("c", 507), _line("d", 518)]
    md = "# Title\n" + "\n".join(lines)
    chunks = chunk_markdown(md, source="test.md")
    assert len(chunks) > 1
    _assert_whole_source_lines(md, chunks)
    # Every chunk that holds source lines ends on a whole one: no fragment.
    assert all(c.content.split("\n")[-1] in [*lines, "# Title"] for c in chunks)


def test_carry_after_paragraph_split_is_dropped_when_next_line_is_long() -> None:
    """A paragraph split carries its last lines before it knows the next line.
    When carry plus that line would reach the cap, the carry is shed up front,
    so the carried line is not emitted again as a chunk of its own."""
    a, b, c, d = _line("a", 700), _line("b", 794), _line("c", 900), _line("d", 100)
    md = f"# T\n{a}\n{b}\n\n{c}\n{d}"
    chunks = chunk_markdown(md, source="test.md")
    _assert_whole_source_lines(md, chunks)
    assert [ch.content for ch in chunks] == [f"# T\n{a}\n{b}", f"{c}\n{d}"]
    # The range starts at the blank line kept from the carry; strip hides it.
    assert chunks[1].start_line == 4


def test_chunks_are_whole_source_lines_on_random_documents() -> None:
    rng = random.Random(956)
    for _ in range(300):
        src: list[str] = []
        for n in range(rng.randint(5, 40)):
            kind = rng.random()
            if kind < 0.15:
                src.append("")
            elif kind < 0.2:
                src.append(f"## H{n}")
            else:
                src.append(_line(f"L{n}", rng.randint(8, 1499)))
        md = "\n".join(src)
        _assert_whole_source_lines(md, chunk_markdown(md, source="test.md"))


def test_single_line_over_cap_is_still_split() -> None:
    long_line = ("word " * 400).strip()
    md = f"# Title\n{_line('a', 300)}\n{long_line}\n{_line('b', 300)}"
    chunks = chunk_markdown(md, source="test.md")
    assert len(long_line) > 1500
    assert all(len(c.content) <= 1500 for c in chunks)
    assert any(c.content.startswith("word") for c in chunks)
    assert sum(c.content.count("word") for c in chunks) >= 400


def test_overlap_lines_shared_between_consecutive_chunks() -> None:
    src = [_line(f"n{n:03d}", 100) for n in range(60)]
    chunks = chunk_markdown("\n".join(src), source="test.md", overlap_lines=2)
    assert len(chunks) > 2
    for prev, nxt in pairwise(chunks):
        assert prev.content.split("\n")[-2:] == nxt.content.split("\n")[:2]
