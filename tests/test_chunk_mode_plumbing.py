"""The chunk mode reaches ``chunk_markdown`` from the constructor and from ``index --chunk-mode``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from click.testing import CliRunner

from memsearch import cli as cli_module
from memsearch import core as core_module
from memsearch.cli import cli
from memsearch.config import MemSearchConfig


class _Embedder:
    model_name = "fake"
    dimension = 4
    batch_size = 8


class _Store:
    def __init__(self, **_kwargs) -> None:
        pass

    def hashes_by_source(self, _source: str) -> set[str]:
        return set()

    def close(self) -> None:
        pass


def _make(monkeypatch, tmp_path: Path, **kwargs):
    seen: list[dict] = []

    def fake_chunk_markdown(text, source="", **kw):
        seen.append(kw)
        return []

    monkeypatch.setattr(core_module, "get_provider", lambda *_a, **_k: _Embedder())
    monkeypatch.setattr(core_module, "MilvusStore", _Store)
    monkeypatch.setattr(core_module, "chunk_markdown", fake_chunk_markdown)
    note = tmp_path / "note.md"
    note.write_text("# Fruit\n\n- apple\n", encoding="utf-8")
    ms = core_module.MemSearch(milvus_uri=str(tmp_path / "x.db"), **kwargs)
    asyncio.run(ms.index_file(note))
    return seen


def test_constructor_passes_block_mode_to_chunk_markdown(monkeypatch, tmp_path: Path) -> None:
    seen = _make(monkeypatch, tmp_path, chunk_mode="block")
    assert [kw["mode"] for kw in seen] == ["block"]


def test_constructor_default_is_section_mode(monkeypatch, tmp_path: Path) -> None:
    seen = _make(monkeypatch, tmp_path)
    assert [kw["mode"] for kw in seen] == ["section"]


class _RecordingMemSearch:
    instances: ClassVar[list[_RecordingMemSearch]] = []

    def __init__(self, paths=None, **kwargs) -> None:
        self.kwargs = kwargs
        self.last_index_stats = {"embedded": 0, "rekeyed": 0}
        _RecordingMemSearch.instances.append(self)

    async def index(self, *, force: bool = False) -> int:
        return 0

    def close(self) -> None:
        pass


def _run_index(monkeypatch, tmp_path: Path, *flags: str):
    _RecordingMemSearch.instances = []
    monkeypatch.setattr(core_module, "MemSearch", _RecordingMemSearch)
    monkeypatch.setattr(cli_module, "resolve_config", lambda _overrides=None: MemSearchConfig())
    result = CliRunner().invoke(cli, ["index", str(tmp_path), *flags])
    return result, _RecordingMemSearch.instances


def test_index_accepts_chunk_mode_block(monkeypatch, tmp_path: Path) -> None:
    result, instances = _run_index(monkeypatch, tmp_path, "--chunk-mode", "block")
    assert result.exit_code == 0, result.output
    assert [i.kwargs["chunk_mode"] for i in instances] == ["block"]


def test_index_rejects_an_unknown_chunk_mode(monkeypatch, tmp_path: Path) -> None:
    result, instances = _run_index(monkeypatch, tmp_path, "--chunk-mode", "nope")
    assert result.exit_code == 2
    assert "nope" in result.output
    assert instances == []


def test_index_without_the_flag_uses_section_mode(monkeypatch, tmp_path: Path) -> None:
    result, instances = _run_index(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    assert [i.kwargs["chunk_mode"] for i in instances] == ["section"]


def test_other_commands_do_not_set_a_chunk_mode(monkeypatch, tmp_path: Path) -> None:
    kwargs = cli_module._cfg_to_memsearch_kwargs(MemSearchConfig())
    assert "chunk_mode" not in kwargs
