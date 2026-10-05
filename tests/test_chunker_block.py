"""Tests for the opt-in ``mode="block"`` chunker."""

import random

import pytest

from memsearch.chunker import _HEADING_RE, _has_meaningful_content, chunk_markdown, clean_content_for_embedding


def _block(text: str, **kw):
    return chunk_markdown(text, source="t.md", mode="block", **kw)


def _contents(chunks):
    return [c.content for c in chunks]


def _spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


# ---------------------------------------------------------------------------
# Block shapes
# ---------------------------------------------------------------------------


def test_each_list_item_is_its_own_chunk():
    chunks = _block("# Fruit\n\n- apple\n- banana\n- cherry\n")
    assert _contents(chunks) == ["# Fruit\n\n- apple", "- banana", "- cherry"]
    assert _spans(chunks) == [(1, 3), (4, 4), (5, 5)]
    assert {(c.heading, c.heading_level) for c in chunks} == {("Fruit", 1)}


def test_list_item_keeps_its_continuation_lines():
    chunks = _block("- apple\n  is red\n  and round\n- banana\n  is long\n\n1. cherry\n2) plum\n")
    assert _contents(chunks) == ["- apple\n  is red\n  and round", "- banana\n  is long", "1. cherry", "2) plum"]
    assert _spans(chunks) == [(1, 3), (4, 5), (7, 7), (8, 8)]


def test_multi_line_paragraph_is_one_chunk():
    chunks = _block("alpha beta\ngamma delta\nepsilon zeta\n\nsecond paragraph\nline two\n")
    assert _contents(chunks) == ["alpha beta\ngamma delta\nepsilon zeta", "second paragraph\nline two"]
    assert _spans(chunks) == [(1, 3), (5, 6)]
    assert chunks[0].heading == "" and chunks[0].heading_level == 0


def test_table_is_one_chunk():
    chunks = _block("# T\n\nrain\nsnow\n| a | b |\n|---|---|\n  | 1 | 2 |\n\nwind gusts\n")
    assert _contents(chunks) == ["# T\n\nrain\nsnow", "| a | b |\n|---|---|\n  | 1 | 2 |", "wind gusts"]
    assert _spans(chunks) == [(1, 4), (5, 7), (9, 9)]


def test_heading_joins_the_first_block_only():
    chunks = _block("## Alpha\nalpha beta\ngamma\n\ndelta epsilon\n")
    assert _contents(chunks) == ["## Alpha\nalpha beta\ngamma", "delta epsilon"]
    assert _spans(chunks) == [(1, 3), (5, 5)]


def test_table_directly_under_heading():
    chunks = _block("# T\n| a | b |\n| 1 | 2 |\n")
    assert _contents(chunks) == ["# T\n| a | b |\n| 1 | 2 |"]
    assert _spans(chunks) == [(1, 3)]


def test_heading_only_section_yields_nothing():
    assert _block("# A\n\n") == []
    chunks = _block("# A\n\n## B\n\nbody text\n\n### C\n")
    assert _contents(chunks) == ["## B\n\nbody text"]
    assert chunks[0].heading == "B" and chunks[0].heading_level == 2


def test_empty_and_blank_input():
    assert _block("") == []
    assert _block("\n\n  \n") == []


def test_no_heading_file_is_a_preamble():
    chunks = _block("alpha\nbeta\n\n- gamma\n")
    assert [(c.heading, c.heading_level) for c in chunks] == [("", 0), ("", 0)]
    assert _contents(chunks) == ["alpha\nbeta", "- gamma"] and _spans(chunks) == [(1, 2), (4, 4)]


def test_text_before_the_first_heading_is_a_preamble():
    chunks = _block("intro words\n\n- one\n\n# Head\n\ntext here\n")
    assert [(c.heading, c.heading_level, c.content) for c in chunks] == [
        ("", 0, "intro words\n\n- one"),
        ("Head", 1, "# Head\n\ntext here"),
    ]


def test_nested_list_item_is_its_own_chunk():
    chunks = _block("- fruit\n  - apple\n    - green\n  - pear\n- weather\n")
    assert _contents(chunks) == ["- fruit", "- apple", "- green", "- pear", "- weather"]


# ---------------------------------------------------------------------------
# Label join
# ---------------------------------------------------------------------------


def test_label_joins_a_following_list_item():
    chunks = _block("Fruits I like:\n\n- apple\n- banana\n")
    assert _contents(chunks) == ["Fruits I like:\n\n- apple", "- banana"]
    assert _spans(chunks) == [(1, 3), (4, 4)]
    chunks = _block("Fruits I like:\n- apple\n- banana\n")
    assert _contents(chunks) == ["Fruits I like:\n- apple", "- banana"]


def test_label_joins_a_following_table():
    chunks = _block("Weather by day:\n\n| day | sky |\n| mon | sun |\n\nafter\n")
    assert _contents(chunks) == ["Weather by day:\n\n| day | sky |\n| mon | sun |", "after"]
    assert _spans(chunks) == [(1, 4), (6, 6)]


def test_label_does_not_join_a_paragraph():
    chunks = _block("Just one line\n\nanother paragraph here\n\n- item\n")
    assert _contents(chunks) == ["Just one line", "another paragraph here\n\n- item"]


def test_label_at_section_end_is_not_joined_across_the_heading():
    chunks = _block("# A\n\nlast words\n\n## B\n\n- item\n")
    assert _contents(chunks) == ["# A\n\nlast words", "## B\n\n- item"]


def test_only_the_second_of_two_one_line_paragraphs_joins_a_list():
    chunks = _block("alpha beta\n\ngamma delta:\n\n- item\n")
    assert _contents(chunks) == ["alpha beta", "gamma delta:\n\n- item"]
    assert _spans(chunks) == [(1, 1), (3, 5)]


def test_label_with_heading_joins_both():
    chunks = _block("# H\n\nLabel:\n\n- one\n- two\n")
    assert _contents(chunks) == ["# H\n\nLabel:\n\n- one", "- two"]


def test_two_line_paragraph_is_not_a_label():
    chunks = _block("line one\nline two\n- item\n")
    assert _contents(chunks) == ["line one\nline two", "- item"]


# ---------------------------------------------------------------------------
# Fences
# ---------------------------------------------------------------------------


def test_fenced_block_with_blank_and_list_lines_is_one_chunk():
    md = "# Code\n\n```\n- not an item\n\nstill code\n```\n\n- real item\n"
    chunks = _block(md)
    assert _contents(chunks) == ["# Code\n\n```\n- not an item\n\nstill code\n```", "- real item"]
    assert _spans(chunks) == [(1, 7), (9, 9)]


def test_tilde_fence_and_info_string():
    chunks = _block("~~~text\n- a\n\n- b\n~~~\n- c\n")
    assert _contents(chunks) == ["~~~text\n- a\n\n- b\n~~~", "- c"]


def test_unclosed_fence_runs_to_the_end_of_the_section():
    chunks = _block("## A\n\n```\n- a\n\n- b\n\n## B\n\n- c\n")
    assert _contents(chunks) == ["## A\n\n```\n- a\n\n- b", "## B\n\n- c"]
    assert _spans(chunks) == [(1, 6), (8, 10)]


def test_inline_triple_backticks_do_not_open_a_fence():
    chunks = _block("use ```x``` here\n\n- a\n")
    assert _contents(chunks) == ["use ```x``` here\n\n- a"]  # one-line paragraph joined as a label


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------


def _paragraph(n: int) -> list[str]:
    return [f"line {i:02d} apple banana cherry" for i in range(n)]


def test_over_cap_paragraph_is_split_into_whole_line_pieces():
    lines = _paragraph(12)  # 26 chars per line
    md = "\n".join(lines)
    chunks = _block(md, max_chunk_size=100)
    assert len(chunks) > 1
    assert all(len(c.content) <= 100 for c in chunks)
    assert "\n".join(c.content for c in chunks) == md
    for c in chunks:
        assert c.content == "\n".join(lines[c.start_line - 1 : c.end_line])
    ends = [c.end_line for c in chunks]
    starts = [c.start_line for c in chunks]
    assert starts[1:] == [e + 1 for e in ends[:-1]]  # no overlap, no gap


def test_block_whose_first_line_is_the_heading_is_capped_too():
    lines = _paragraph(8)
    chunks = _block("## Head\n" + "\n".join(lines), max_chunk_size=100)
    assert len(chunks) > 1
    assert chunks[0].content.startswith("## Head\nline 00")
    assert all(len(c.content) <= 100 for c in chunks)
    assert all(c.heading == "Head" for c in chunks)
    assert "\n".join(c.content for c in chunks) == "## Head\n" + "\n".join(lines)


def test_cap_boundary_agrees_with_section_mode():
    # Section mode keeps a section of exactly max_chunk_size characters whole
    # and splits one character more; a block does the same.
    exact = "a" * 9 + "\n" + "b" * 10  # 20 chars
    assert len(exact) == 20
    assert _contents(_block(exact, max_chunk_size=20)) == [exact]
    assert _contents(chunk_markdown(exact, mode="section", max_chunk_size=20)) == [exact]
    over = exact + "c"
    assert _contents(_block(over, max_chunk_size=20)) == ["a" * 9, "b" * 10 + "c"]
    assert len(chunk_markdown(over, max_chunk_size=20)) > 1


def test_single_over_cap_line_is_cut_as_section_mode_cuts_it():
    sentence = "Apples fall in autumn. Rain comes in spring. Snow is rare here. "
    line = sentence * 5
    section = chunk_markdown(line, max_chunk_size=100)
    block = _block(line, max_chunk_size=100)
    assert len(block) > 1
    assert _contents(block) == _contents(section)
    assert _spans(block) == _spans(section) == [(1, 1)] * len(block)


def test_over_cap_line_inside_a_block_splits_the_block_around_it():
    long_line = "word " * 40  # 200 chars
    md = "alpha beta\n" + long_line.strip() + "\ngamma delta"
    chunks = _block(md, max_chunk_size=100)
    assert _contents(chunks)[0] == "alpha beta"
    assert _contents(chunks)[-1] == "gamma delta"
    assert all(len(c.content) <= 100 for c in chunks)
    assert "".join("".join(c.content.split()) for c in chunks[1:-1]) == "word" * 40


def test_overlap_lines_is_ignored():
    md = "# H\n\n" + "\n".join(_paragraph(12)) + "\n\n- item\n- other\n"
    base = _block(md, max_chunk_size=100, overlap_lines=0)
    assert base == _block(md, max_chunk_size=100, overlap_lines=5)
    seen: list[int] = []
    for c in base:
        seen.extend(range(c.start_line, c.end_line + 1))
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# Line endings and modes
# ---------------------------------------------------------------------------


def test_crlf_text_chunks_like_lf_text():
    md = "# H\n\nLabel:\n\n- one\n  more\n- two\n\n| a | b |\n\n```\n- x\n\ny\n```\n\n## Empty\n"
    assert _block(md.replace("\n", "\r\n")) == _block(md)
    assert "\r" not in "".join(_contents(_block(md.replace("\n", "\r\n"))))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="bogus"):
        chunk_markdown("# A\n\ntext\n", mode="bogus")
    with pytest.raises(ValueError):
        chunk_markdown("", mode="")


def test_default_mode_is_section():
    md = "# A\n\n- one\n- two\n\n## B\n\ntext\n"
    assert chunk_markdown(md) == chunk_markdown(md, mode="section")
    assert len(chunk_markdown(md)) == 2
    assert len(chunk_markdown(md, mode="block")) == 3


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

_WORDS = ["apple", "banana", "cherry", "plum", "rain", "snow", "wind", "sun", "alpha", "beta", "gamma", "delta"]
_N_DOCS = 2500


def _words(rng: random.Random, lo: int = 1, hi: int = 6) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(lo, hi)))


def _gen_doc(seed: int, cap: int) -> str:
    """A seeded random markdown document exercising every block shape."""
    rng = random.Random(seed)
    out: list[str] = []
    headed = rng.random() > 0.15
    unclosed_fence_at_end = rng.random() < 0.05

    def list_items() -> None:
        for _ in range(rng.randint(1, 4)):
            marker = rng.choice(["- ", "* ", "+ ", "1. ", "2) "])
            indent = rng.choice(["", "", "  ", "    "])
            out.append(f"{indent}{marker}{_words(rng)}")
            out.extend(f"{indent}  {_words(rng)}" for _ in range(rng.choice([0, 0, 1, 2])))

    def table() -> None:
        out.extend(
            rng.choice(["| ", "  | "]) + " | ".join(_words(rng, 1, 2) for _ in range(2)) + " |"
            for _ in range(rng.randint(1, 4))
        )

    def paragraph() -> None:
        out.extend(_words(rng, 1, 8) for _ in range(rng.choice([1, 1, 2, 3, 4])))

    def fence() -> None:
        out.append("```")
        out.extend(
            rng.choice(["", "- " + _words(rng), _words(rng), "| " + _words(rng, 1, 2), ""])
            for _ in range(rng.randint(1, 5))
        )
        out.append("```")

    def long_line() -> None:
        text = _words(rng, 3, 6)
        while len(text) < cap + rng.randint(0, cap):
            text += " " + _words(rng, 3, 6)
        out.append(text + rng.choice(["", ".", ". " + _words(rng)]))

    def label_then() -> None:
        out.append(_words(rng, 1, 3) + ":")
        if rng.random() < 0.5:
            out.append("")
        rng.choice([list_items, table])()

    emitters = [list_items, list_items, table, paragraph, paragraph, fence, long_line, label_then, label_then]
    for _ in range(rng.randint(1, 12)):
        if headed and rng.random() < 0.35:
            out.append("#" * rng.randint(1, 4) + " " + _words(rng, 1, 3))
            if rng.random() < 0.3:
                table()  # a table directly under the heading
                continue
        elif rng.random() < 0.1:
            out.extend([""] * rng.randint(1, 3))
        rng.choice(emitters)()
        if rng.random() < 0.6:
            out.extend([""] * rng.choice([1, 1, 1, 2, 3]))
    if unclosed_fence_at_end:
        out.extend(["```", "- " + _words(rng), "", _words(rng)])
    text = "\n".join(out)
    if headed and not text.lstrip().startswith("#") and rng.random() < 0.5:
        text = "preamble " + _words(rng) + "\n\n" + text
    return text


def _sections(lines: list[str]) -> list[tuple[int, int]]:
    """Section ranges that section mode keeps, found independently of the chunker."""
    heads = [i for i, ln in enumerate(lines) if _HEADING_RE.match(ln)]
    bounds = ([0] if not heads or heads[0] > 0 else []) + heads
    ranges = [(b, bounds[k + 1] if k + 1 < len(bounds) else len(lines)) for k, b in enumerate(bounds)]
    kept = []
    for s, e in ranges:
        text = "\n".join(lines[s:e]).strip()
        if text and _has_meaningful_content(text):
            kept.append((s, e))
    return kept


def _fence_flags(lines: list[str]) -> list[bool]:
    """flags[i] is True when line i lies strictly inside a ``` fence pair (or an unclosed one)."""
    flags, inside = [], False
    for ln in lines:
        if ln.strip() == "```":
            flags.append(False)
            inside = not inside
        else:
            flags.append(inside)
    return flags


def _is_item_or_row(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("|") or s[:2] in ("- ", "* ", "+ ") or (s[:1].isdigit() and s[1:3] in (". ", ") "))


def _check_doc(doc: str, cap: int) -> None:
    lines = doc.split("\n")
    chunks = chunk_markdown(doc, source="p.md", mode="block", max_chunk_size=cap)
    flags = _fence_flags(lines)
    stripped = {ln.strip() for ln in lines}

    def is_piece(c) -> bool:
        return c.start_line == c.end_line and len(lines[c.start_line - 1]) >= cap

    for c in chunks:
        # (3) the cap: never exceeded, because a long line is cut by the section-mode routine
        assert len(c.content) <= cap, (doc, c)
        # (5) something is left to embed
        assert clean_content_for_embedding(c.content), (doc, c)
        if is_piece(c):
            assert "".join(c.content.split()) in "".join(lines[c.start_line - 1].split())
            continue
        # (1) every line is a source line
        assert all(ln.strip() in stripped for ln in c.content.split("\n")), (doc, c)
        # (4) the line range reproduces the content
        assert c.content == "\n".join(lines[c.start_line - 1 : c.end_line]).strip(), (doc, c)
        # (6) blank lines only after a heading, after a joined label, or inside a fence
        src = list(range(c.start_line - 1, c.end_line))
        for pos, k in enumerate(src):
            if lines[k].strip() or flags[k]:
                continue
            before = [lines[j] for j in src[:pos] if lines[j].strip()]
            after = next((lines[j] for j in src[pos:] if lines[j].strip()), "")
            heading_first = bool(before) and bool(_HEADING_RE.match(before[0]))
            after_heading = heading_first and len(before) == 1
            label_only = len(before) == (2 if heading_first else 1)
            after_label = label_only and not _is_item_or_row(before[-1]) and _is_item_or_row(after)
            assert after_heading or after_label, (doc, c, k)

    # (2) each kept section is covered once, in order
    for s, e in _sections(lines):
        expected: list = [
            ("OC", i, "".join(lines[i].split())) if len(lines[i]) >= cap else lines[i].strip()
            for i in range(s, e)
            if lines[i].strip()
        ]
        actual: list = []
        for c in chunks:
            if not (s <= c.start_line - 1 < e):
                continue
            assert c.end_line <= e, (doc, c)
            if is_piece(c):
                i = c.start_line - 1
                piece = "".join(c.content.split())
                if actual and isinstance(actual[-1], tuple) and actual[-1][1] == i:
                    actual[-1] = ("OC", i, actual[-1][2] + piece)
                else:
                    actual.append(("OC", i, piece))
            else:
                actual.extend(ln.strip() for ln in c.content.split("\n") if ln.strip())
        assert actual == expected, (doc, s, e)

    # (7) CRLF text chunks exactly like LF text
    assert chunk_markdown(doc.replace("\n", "\r\n"), source="p.md", mode="block", max_chunk_size=cap) == chunks


def test_block_mode_properties_on_random_documents():
    seen_chunks = 0
    for seed in range(_N_DOCS):
        cap = (80, 200, 1500)[seed % 3]
        doc = _gen_doc(seed, cap)
        _check_doc(doc, cap)
        seen_chunks += len(chunk_markdown(doc, mode="block", max_chunk_size=cap))
    assert seen_chunks > _N_DOCS * 3


def test_generator_reaches_every_shape():
    docs = [_gen_doc(s, (80, 200, 1500)[s % 3]) for s in range(_N_DOCS)]
    joined = "\n".join(docs)
    assert any(not _HEADING_RE.search(d) for d in docs)
    assert any("\n```\n" in d and "\n\n" in d.split("```")[1] for d in docs)
    assert any(len(ln) >= 80 for d in docs for ln in d.split("\n"))
    assert "\n    - " in joined and "\n| " in joined and "\n\n\n" in joined
    assert any(d.count("```") % 2 for d in docs)  # an unclosed fence
