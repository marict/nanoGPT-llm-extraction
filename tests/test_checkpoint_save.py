import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

train = importlib.import_module("train")


def test_safe_torch_save_retry(monkeypatch, tmp_path):
    calls = {"count": 0}

    def flaky_save(obj, path, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("unexpected pos 10 vs 9")
        Path(path).write_text("ok")

    monkeypatch.setattr(train, "_st", SimpleNamespace(save_file=flaky_save))

    out_path = tmp_path / "ckpt.pt"
    train._safe_torch_save({"a": 1}, out_path, retries=1)
    assert out_path.exists()
    assert calls["count"] == 2


def test_safe_torch_save_failure(monkeypatch, tmp_path):
    def always_fail(obj, path, **kwargs):
        raise RuntimeError("disk full?")

    monkeypatch.setattr(train, "_st", SimpleNamespace(save_file=always_fail))
    with pytest.raises(train.CheckpointSaveError):
        train._safe_torch_save({"a": 1}, tmp_path / "ckpt.pt", retries=1)
