import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)

from dag_model import (GPT, MLP, Block, CausalSelfAttention, GPTConfig,
                       LayerNorm)


@pytest.fixture(scope="module")
def small_gpt():
    config = GPTConfig(
        vocab_size=10, block_size=4, n_layer=2, n_head=2, n_embd=8, dag_depth=0
    )
    return GPT(config)


def test_model_forward(small_gpt):
    model = small_gpt
    x = torch.randint(0, 10, (1, 4))
    out, _ = model(x, targets=x)
    assert out.shape == (1, 4, 10)


def test_causal_self_attention():
    # Test CausalSelfAttention with a simple input
    config = GPTConfig(n_embd=4, n_head=2, block_size=8, dag_depth=0)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 8, 4)
    out = attn(x)
    assert out.shape == x.shape, "Attention output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(
        out, torch.zeros_like(out)
    ), "Attention output is all zeros"


def test_layer_norm():
    # Test LayerNorm with a simple input
    layer_norm = LayerNorm(4, bias=True)
    x = torch.randn(2, 8, 4)
    out = layer_norm(x)
    assert out.shape == x.shape, "LayerNorm output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(
        out, torch.zeros_like(out)
    ), "LayerNorm output is all zeros"


def test_mlp():
    # Test MLP with a simple input
    config = GPTConfig(n_embd=4, dag_depth=0)
    mlp = MLP(config)
    x = torch.randn(2, 8, 4)
    out = mlp(x)
    assert out.shape == x.shape, "MLP output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out)), "MLP output is all zeros"


def test_block():
    # Test Block with a simple input
    config = GPTConfig(n_embd=4, n_head=2, block_size=8, dag_depth=0)
    block = Block(config)
    x = torch.randn(2, 8, 4)
    out = block(x)
    assert out.shape == x.shape, "Block output shape mismatch"
    # Check that the output is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out)), "Block output is all zeros"


def test_extra_vals_gpt():
    """Test that GPT's extra_vals returns an empty dict."""
    config = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=0
    )
    model = GPT(config)

    # Test that extra_vals returns empty dict before any forward pass
    extra_vals = model.extra_vals()
    assert isinstance(extra_vals, dict)
    assert len(extra_vals) == 0

    # Test that extra_vals still returns empty dict after forward pass
    x = torch.randint(0, config.vocab_size, (2, 8))
    model(x)
    extra_vals_after_forward = model.extra_vals()
    assert isinstance(extra_vals_after_forward, dict)
    assert len(extra_vals_after_forward) == 0


def test_gpt_dag_depth_zero():
    """Test that GPT with dag_depth=0 behaves like standard GPT."""
    config = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=0
    )
    model = GPT(config)

    # Should not have DAG-specific components
    assert not hasattr(model, "value_extractor")
    assert not hasattr(model, "dag")
    assert not hasattr(model, "mix_gate")

    # Forward pass should work
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 100)
    assert loss is not None

    # Generate should work
    generated = model.generate(x[:, :4], max_new_tokens=4)
    assert generated.shape == (2, 8)


def test_gpt_dag_depth_nonzero():
    """Test that GPT with dag_depth>0 has DAG components."""
    config = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(config)

    # Should have DAG-specific components
    assert hasattr(model, "value_extractor")
    assert hasattr(model, "dag")
    assert hasattr(model, "mix_gate")

    # Forward pass should work
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 100)
    assert loss is not None

    # Should have node values
    node_values = model.get_node_values_list()
    assert isinstance(node_values, list)
    assert len(node_values) > 0
