import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from model import GPT, GPTConfig


@pytest.fixture(scope="module")
def small_gpt():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=2, n_head=2, n_embd=8)
    return GPT(config)


def test_model_forward(small_gpt):
    model = small_gpt
    x = torch.randint(0, 10, (1, 4))
    out, _ = model(x, targets=x)
    assert out.shape == (1, 4, 10)
