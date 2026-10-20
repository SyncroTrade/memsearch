"""Unit tests for MEMSEARCH_ONNX_THREADS in the ONNX embedding provider.

No model is downloaded: onnxruntime, huggingface_hub and tokenizers are faked.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

_DEFAULT = object()


def _install_fakes(monkeypatch, *, record: dict):
    ort = types.ModuleType("onnxruntime")

    class FakeSessionOptions:
        def __init__(self):
            self.intra_op_num_threads = _DEFAULT
            record["options"] = self

    class FakeSession:
        def __init__(self, path, sess_options=None, **kwargs):
            record["path"] = path
            record["sess_options"] = sess_options
            self._outputs = [types.SimpleNamespace(name="dense_vecs")]

        def get_outputs(self):
            return self._outputs

    ort.SessionOptions = FakeSessionOptions
    ort.InferenceSession = FakeSession

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: "unused"
    hub.list_repo_files = lambda *a, **k: []

    tok = types.ModuleType("tokenizers")

    class FakeTokenizer:
        @staticmethod
        def from_file(path):
            return FakeTokenizer()

        def enable_padding(self, **kwargs):
            pass

        def enable_truncation(self, **kwargs):
            pass

    tok.Tokenizer = FakeTokenizer

    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)


def _build(monkeypatch, record: dict):
    _install_fakes(monkeypatch, record=record)
    sys.modules.pop("memsearch.embeddings.onnx", None)
    mod = importlib.import_module("memsearch.embeddings.onnx")
    monkeypatch.setattr(
        mod.OnnxEmbedding,
        "_download_model_files",
        staticmethod(lambda model, dl, ls: ("tok.json", "model.onnx")),
    )
    monkeypatch.setattr(mod.OnnxEmbedding, "_encode", lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts])
    return mod.OnnxEmbedding()


def test_onnx_threads_env_sets_intra_op_threads(monkeypatch):
    record: dict = {}
    monkeypatch.setenv("MEMSEARCH_ONNX_THREADS", "8")

    provider = _build(monkeypatch, record)

    assert record["sess_options"] is record["options"]
    assert record["sess_options"].intra_op_num_threads == 8
    assert provider.dimension == 3


def test_onnx_threads_unset_keeps_default(monkeypatch):
    record: dict = {}
    monkeypatch.delenv("MEMSEARCH_ONNX_THREADS", raising=False)

    _build(monkeypatch, record)

    assert record["sess_options"] is record["options"]
    assert record["sess_options"].intra_op_num_threads is _DEFAULT


@pytest.mark.parametrize("value", ["abc", "0", "-3", ""])
def test_onnx_threads_invalid_keeps_default(monkeypatch, value):
    record: dict = {}
    monkeypatch.setenv("MEMSEARCH_ONNX_THREADS", value)

    _build(monkeypatch, record)

    assert record["sess_options"] is record["options"]
    assert record["sess_options"].intra_op_num_threads is _DEFAULT
