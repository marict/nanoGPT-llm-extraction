import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_checkpoints as cc


class DummyModel:
    def __call__(self, x, targets=None):
        return None, torch.tensor(0.5)


class DummyEnc:
    def encode(self, text):
        return [0, 1]


def test_evaluate_constant_loss():
    path = Path(__file__).resolve().parent / "math_eval.txt"
    pairs = cc.load_pairs(path)
    enc = DummyEnc()
    model = DummyModel()
    loss = cc.evaluate(model, enc, pairs, "cpu")
    assert loss == pytest.approx(0.5)
