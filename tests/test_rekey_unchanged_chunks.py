"""Tests for #704: re-keying unchanged chunks instead of re-embedding them.

Inserting a new section into a markdown file shifts the line ranges of
every chunk below it. Since a chunk's composite id is derived in part
from its line range, a pure shift produces a brand new id for
byte-identical content -- which the old logic treated as "new content"
and paid a full embed call for. These tests exercise the fix: a shifted
chunk whose content is unchanged should copy its donor row's vector
instead of being re-embedded, while genuinely new or edited content
(and force=True) must still go through the embedder.

Follows the FakeEmbedder/InMemoryStore pattern from test_index_cleanup.py,
extended with an embed-call counter and a ``rows_by_hashes`` fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memsearch.chunker import compute_chunk_id, compute_content_hash
from memsearch.core import MemSearch


class CountingEmbedder:
    """Same shape as FakeEmbedder (test_index_cleanup.py) but counts embed calls."""

    def __init__(self, model_name: str = "fake") -> None:
        self._model_name = model_name
        self.embed_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        # Distinct-ish vector per distinct text so donor-vector reuse is
        # observable (a re-keyed row keeps the OLD vector, never a
        # freshly computed one).
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class InMemoryStore:
    """Same as test_index_cleanup.InMemoryStore, plus rows_by_hashes."""

    def __init__(self) -> None:
        self._records_by_source: dict[str, list[dict[str, Any]]] = {}
        self.deleted_sources: list[str] = []
        self.upsert_calls: int = 0

    def hashes_by_source(self, source: str) -> set[str]:
        return {record["chunk_hash"] for record in self._records_by_source.get(source, [])}

    def delete_by_hashes(self, hashes: list[str]) -> None:
        stale = set(hashes)
        for source, records in list(self._records_by_source.items()):
            remaining = [record for record in records if record["chunk_hash"] not in stale]
            if remaining:
                self._records_by_source[source] = remaining
            else:
                self._records_by_source.pop(source, None)

    def upsert(self, records: list[dict[str, Any]]) -> int:
        self.upsert_calls += 1
        for record in records:
            source = record["source"]
            existing = [
                old for old in self._records_by_source.get(source, []) if old["chunk_hash"] != record["chunk_hash"]
            ]
            existing.append(record)
            self._records_by_source[source] = existing
        return len(records)

    def indexed_sources(self) -> set[str]:
        return set(self._records_by_source)

    def delete_by_source(self, source: str) -> None:
        self.deleted_sources.append(source)
        self._records_by_source.pop(source, None)

    def rows_by_hashes(self, hashes: list[str], *, batch_size: int = 500) -> list[dict[str, Any]]:
        wanted = set(hashes)
        rows: list[dict[str, Any]] = []
        for records in self._records_by_source.values():
            rows.extend(dict(record) for record in records if record["chunk_hash"] in wanted)
        return rows

    def all_records(self) -> list[dict[str, Any]]:
        return [r for records in self._records_by_source.values() for r in records]


def make_memsearch(paths: list[str | Path], embedder: CountingEmbedder) -> tuple[MemSearch, InMemoryStore]:
    ms = MemSearch.__new__(MemSearch)
    ms._paths = [str(p) for p in paths]
    ms._max_chunk_size = 1500
    ms._overlap_lines = 2
    ms._chunk_mode = "section"
    ms._embedder = embedder
    store = InMemoryStore()
    ms._store = store
    ms._reranker_model = ""
    ms.last_index_stats = {"embedded": 0, "rekeyed": 0}
    return ms, store


def write_note(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_V0 = (
    "# Section A\n\ncontent aaaa aaaa aaaa\n\n"
    "# Section B\n\ncontent bbbb bbbb bbbb\n\n"
    "# Section C\n\ncontent cccc cccc cccc\n"
)

_V1_INSERT_AT_TOP = (
    "# Section NEW\n\nbrand new content zzzz zzzz\n\n"
    "# Section A\n\ncontent aaaa aaaa aaaa\n\n"
    "# Section B\n\ncontent bbbb bbbb bbbb\n\n"
    "# Section C\n\ncontent cccc cccc cccc\n"
)


@pytest.mark.asyncio
async def test_insert_at_top_rekeys_shifted_chunks_instead_of_reembedding(tmp_path: Path) -> None:
    f = write_note(tmp_path / "doc.md", _V0)
    embedder = CountingEmbedder()
    ms, store = make_memsearch([f], embedder)

    n1 = await ms.index()
    assert n1 == 3
    old_ids = store.hashes_by_source(str(f))
    assert len(old_ids) == 3

    write_note(f, _V1_INSERT_AT_TOP)
    embedder.embed_calls.clear()
    n2 = await ms.index()

    # 4 chunks written total, but only the brand-new section was embedded.
    assert n2 == 4
    embedded_texts = [t for call in embedder.embed_calls for t in call]
    assert len(embedded_texts) == 1
    assert "brand new content" in embedded_texts[0]
    assert ms.last_index_stats == {"embedded": 1, "rekeyed": 3}

    new_ids = store.hashes_by_source(str(f))
    assert len(new_ids) == 4
    # Every old id is gone -- shifted chunks moved to a NEW composite id.
    assert old_ids.isdisjoint(new_ids)

    # The re-keyed rows kept their ORIGINAL vectors (donor vectors), not a
    # freshly computed one: every non-new record's vector[0] equals
    # len(content) because CountingEmbedder derives it deterministically
    # from the text, so a mismatch would mean it was silently re-embedded.
    for record in store.all_records():
        if "brand new content" in record["content"]:
            continue
        assert record["embedding"][0] == float(len(record["content"].strip()))


@pytest.mark.asyncio
async def test_model_mismatch_donor_is_not_reused(tmp_path: Path) -> None:
    f = write_note(tmp_path / "doc.md", _V0)
    embedder = CountingEmbedder(model_name="model-a")
    ms, store = make_memsearch([f], embedder)
    await ms.index()

    # Tamper: rewrite every stored row's chunk_hash as if it had been minted
    # under a DIFFERENT model. These rows now fail the integrity check.
    for records in store._records_by_source.values():
        for record in records:
            content_hash = compute_content_hash(record["content"])
            record["chunk_hash"] = compute_chunk_id(
                record["source"], record["start_line"], record["end_line"], content_hash, "some-other-model"
            )

    write_note(f, _V1_INSERT_AT_TOP)
    embedder.embed_calls.clear()
    n2 = await ms.index()

    assert n2 == 4
    # No valid donors under "model-a" -- every chunk (new + shifted) is
    # re-embedded, exactly like the old behavior.
    embedded_texts = [t for call in embedder.embed_calls for t in call]
    assert len(embedded_texts) == 4
    assert ms.last_index_stats == {"embedded": 4, "rekeyed": 0}


@pytest.mark.asyncio
async def test_force_reembeds_everything_even_with_available_donors(tmp_path: Path) -> None:
    f = write_note(tmp_path / "doc.md", _V0)
    embedder = CountingEmbedder()
    ms, _store = make_memsearch([f], embedder)
    await ms.index()

    write_note(f, _V1_INSERT_AT_TOP)
    embedder.embed_calls.clear()
    n2 = await ms.index(force=True)

    assert n2 == 4
    embedded_texts = [t for call in embedder.embed_calls for t in call]
    assert len(embedded_texts) == 4
    assert ms.last_index_stats == {"embedded": 4, "rekeyed": 0}


@pytest.mark.asyncio
async def test_content_edit_is_embedded_not_rekeyed(tmp_path: Path) -> None:
    f = write_note(tmp_path / "doc.md", _V0)
    embedder = CountingEmbedder()
    ms, _store = make_memsearch([f], embedder)
    await ms.index()

    # Edit Section B's body in place -- same heading, same line range,
    # different content_hash. This must NOT be treated as a re-key: it
    # is genuinely new content and has no donor.
    edited = (
        "# Section A\n\ncontent aaaa aaaa aaaa\n\n"
        "# Section B\n\nthis body was edited\n\n"
        "# Section C\n\ncontent cccc cccc cccc\n"
    )
    write_note(f, edited)
    embedder.embed_calls.clear()
    n2 = await ms.index()

    assert n2 == 1
    embedded_texts = [t for call in embedder.embed_calls for t in call]
    assert len(embedded_texts) == 1
    assert "this body was edited" in embedded_texts[0]
    assert ms.last_index_stats == {"embedded": 1, "rekeyed": 0}
