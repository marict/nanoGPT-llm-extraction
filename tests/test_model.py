import os
import sys
import pytest
torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import GPT, GPTConfig


def test_model_forward():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=2, n_head=2, n_embd=8)
    model = GPT(config)
    x = torch.randint(0, 10, (1, 4))
    out, _ = model(x, targets=x)
    assert out.shape == (1, 4, 10)
