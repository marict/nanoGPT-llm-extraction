import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from model import GPT, MLP, Block, CausalSelfAttention, GPTConfig, LayerNorm


@pytest.fixture(scope="module")
def small_gpt():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=2, n_head=2, n_embd=8)
    return GPT(config)


def test_model_forward(small_gpt):
    model = small_gpt
    x = torch.randint(0, 10, (1, 4))
    out, _ = model(x, targets=x)
    assert out.shape == (1, 4, 10)


def test_layernorm():
    # Test LayerNorm with a simple input
    x = torch.randn(2, 3, 4)
    ln = LayerNorm(4, bias=True)
    out = ln(x)
    assert out.shape == x.shape, "LayerNorm output shape mismatch"
    # Check that the output is normalized (mean close to 0, std close to 1)
    assert torch.allclose(
        out.mean(dim=-1), torch.zeros_like(out.mean(dim=-1)), atol=1e-5
    ), "LayerNorm mean not close to 0"
    assert torch.allclose(
        out.std(dim=-1), torch.ones_like(out.std(dim=-1)), atol=0.2
    ), "LayerNorm std not close to 1"


def test_causal_self_attention():
    # Test CausalSelfAttention with a simple input
    config = GPTConfig(n_embd=4, n_head=2, block_size=8)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 8, 4)
    out = attn(x)
    assert out.shape == x.shape, "CausalSelfAttention output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(
        out, torch.zeros_like(out)
    ), "CausalSelfAttention output is all zeros"


def test_mlp():
    # Test MLP with a simple input
    config = GPTConfig(n_embd=4)
    mlp = MLP(config)
    x = torch.randn(2, 8, 4)
    out = mlp(x)
    assert out.shape == x.shape, "MLP output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out)), "MLP output is all zeros"


def test_block():
    # Test Block with a simple input
    config = GPTConfig(n_embd=4, n_head=2, block_size=8)
    block = Block(config)
    x = torch.randn(2, 8, 4)
    out = block(x)
    assert out.shape == x.shape, "Block output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out)), "Block output is all zeros"


def test_extra_vals_gpt():
    """Test that GPT's extra_vals returns an empty dict."""
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100)
    model = GPT(config)
    extra_vals = model.extra_vals()
    assert isinstance(extra_vals, dict)
    assert len(extra_vals) == 0
