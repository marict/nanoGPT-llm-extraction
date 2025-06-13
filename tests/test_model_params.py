import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import GPT, GPTConfig


def test_parameter_count_positive():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT(config)
    count = model.get_num_params()
    assert isinstance(count, int)
    assert count > 0
