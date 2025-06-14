import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import GPT, GPTConfig
import pytest


@pytest.fixture(scope="module")
def simple_model():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8)
    return GPT(config)


def test_parameter_count_positive(simple_model):
    model = simple_model
    count = model.get_num_params()
    assert isinstance(count, int)
    assert count > 0
