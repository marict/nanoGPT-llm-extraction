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


def test_model_forward_and_generation(small_gpt):
    """Test model forward pass, generation, and parameter counting."""
    model = small_gpt
    x = torch.randint(0, 10, (1, 4))

    # Test forward pass
    out, _ = model(x, targets=x)
    assert out.shape == (1, 4, 10)

    # Test generation capability
    generated = model.generate(x, max_new_tokens=2)
    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :4], x)
    assert torch.all(generated < 10)  # Within vocab size

    # Test parameter counting
    param_count = model.get_num_params()
    assert isinstance(param_count, int)
    assert param_count > 0


def test_transformer_components():
    """Test all transformer component shapes and basic functionality."""
    config = GPTConfig(n_embd=4, n_head=2, block_size=8, dag_depth=0)
    x = torch.randn(2, 8, 4)

    # Test CausalSelfAttention
    attn = CausalSelfAttention(config)
    attn_out = attn(x)
    assert attn_out.shape == x.shape

    # Test LayerNorm
    layer_norm = LayerNorm(4, bias=True)
    ln_out = layer_norm(x)
    assert ln_out.shape == x.shape

    # Test MLP
    mlp = MLP(config)
    mlp_out = mlp(x)
    assert mlp_out.shape == x.shape

    # Test Block (combines attention + MLP)
    block = Block(config)
    block_out = block(x)
    assert block_out.shape == x.shape


def test_gpt_dag_depth_variations():
    """Test GPT behavior with different DAG depths."""
    base_config = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100
    )

    # Test DAG depth = 0 (standard GPT)
    config_zero = GPTConfig(**{**base_config.__dict__, "dag_depth": 0})
    model_zero = GPT(config_zero)

    # Should not have DAG-specific components
    assert not hasattr(model_zero, "dag")  # No DAG components when dag_depth=0
    assert not hasattr(model_zero, "dag")
    assert not hasattr(model_zero, "dag_mixer")  # No DAG mixer when dag_depth=0

    # Test DAG depth > 0 (DAG-enabled GPT)
    config_dag = GPTConfig(**{**base_config.__dict__, "dag_depth": 2})
    model_dag = GPT(config_dag)

    # Should have DAG-specific components
    assert hasattr(model_dag.dag, "value_extractor")  # Value extractor now in DAG
    assert hasattr(model_dag, "dag")
    assert hasattr(model_dag, "dag_mixer")  # DAG mixer present when dag_depth>0

    # Test forward pass works for both
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))

    for model in [model_zero, model_dag]:
        logits, loss = model(x, y)
        assert logits.shape == (2, 8, 100)
        assert loss is not None

        # Test generation
        generated = model.generate(x[:, :4], max_new_tokens=4)
        assert generated.shape == (2, 8)


def test_extra_vals_functionality():
    """Test that standard GPT models work without extra logging."""
    # Standard GPT should work fine without extra_vals method
    config_std = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=0
    )
    model_std = GPT(config_std)

    # Model should work fine without extra_vals
    x = torch.randint(0, 100, (2, 8))
    logits, loss = model_std(x)
    assert logits is not None

    # DAG model should have node values
    config_dag = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model_dag = GPT(config_dag)
    model_dag(x)

    # Use DAGLogger to get node values
    from dag_logger import DAGLogger

    logger = DAGLogger()
    node_values = logger.get_node_values_list(model_dag)
    assert isinstance(node_values, list)
    assert len(node_values) > 0
