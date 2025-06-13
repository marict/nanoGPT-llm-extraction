import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import compare_checkpoints as cc


class DummyModel:
    def __call__(self, x, targets=None):
        return None, torch.tensor(0.5)


class DummyEnc:
    def encode(self, text):
        return [0, 1]


def test_evaluate_constant_loss():
    path = os.path.join(os.path.dirname(__file__), "math_eval.txt")
    pairs = cc.load_pairs(path)
    enc = DummyEnc()
    model = DummyModel()
    loss = cc.evaluate(model, enc, pairs, "cpu")
    assert loss == pytest.approx(0.5)
