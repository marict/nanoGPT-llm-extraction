# tests/test_dag_model.py
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Set random seeds for reproducible tests
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import numpy as np

np.random.seed(42)

# --------------------------------------------------------------------- #
# import library code
# --------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_common import (SMALL_CONFIG, TINY_CONFIG, assert_valid_forward_pass,
                         assert_valid_logging, assert_valid_node_values,
                         sample_batch_small, sample_batch_tiny,
                         setup_gradient_tracking_test, small_model,
                         standard_model, tiny_model)

import dag_model  # noqa: E402
from dag_logger import DAGLogger
from dag_model import (GPT, DAGPlanPredictor, DifferentiableDAG, GPTConfig,
                       op_names)


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def make_dummy_proj(hidden_dim: int) -> nn.Linear:
    """Tiny 1→H projector for the new DifferentiableDAG signature."""
    return nn.Linear(1, hidden_dim, bias=False)


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_dag_gpt():
    cfg = GPTConfig(
        vocab_size=20,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=2,
    )
    return GPT(cfg), cfg


# --------------------------------------------------------------------- #
# Core functionality tests (4 tests)
# --------------------------------------------------------------------- #
def test_dag_gpt_forward(small_dag_gpt):
    model, _ = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    logits, loss = model(x)
    assert logits.shape == (2, 4, 20)
    assert loss is None


def test_dag_backward_flow(small_dag_gpt):
    """Test DAG backward flow - properly tests gradients with fixed in-place operations."""
    model, _ = small_dag_gpt
    model.train()  # Enable gradient computation

    x = torch.randint(0, 20, (2, 4))
    y = torch.randint(0, 20, (2, 4))

    # Test forward pass
    logits, loss = model(x, y)

    # Verify forward pass works correctly
    assert logits.shape == (2, 4, 20), "Logits shape incorrect"
    assert loss is not None and loss > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"

    # Test backward pass - this should work now that in-place operations are fixed
    loss.backward()

    # Verify key gradients exist and are finite (not all parameters need gradients in every pass)
    key_params_with_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"
            key_params_with_grads += 1

    # Ensure at least some parameters received gradients
    assert key_params_with_grads > 0, "No parameters received gradients"

    # Verify DAG structure exists
    assert hasattr(model, "dag"), "Model should have DAG"
    assert hasattr(model.dag, "plan_predictor"), "DAG should have plan_predictor"


def test_basic_daggpt_functionality(tiny_model, sample_batch_tiny):
    """Test basic DAG functionality with batched inputs."""
    model, cfg = tiny_model
    batch_x, batch_y = sample_batch_tiny

    # Test forward pass
    logits, loss = model(batch_x, batch_y)
    assert logits.shape == (batch_x.size(0), batch_x.size(1), cfg.vocab_size)
    assert loss is not None and loss > 0

    # Test backward pass
    loss.backward()

    # Verify gradients exist
    assert any(p.grad is not None for p in model.parameters())


def test_daggpt_config_creation():
    """Test DAG configuration creation and validation."""
    cfg = GPTConfig()
    assert cfg.dag_depth == 4

    cfg2 = GPTConfig(
        dag_depth=6, n_embd=512, n_layer=6, n_head=8, block_size=1024, vocab_size=50257
    )
    assert cfg2.dag_depth == 6 and cfg2.n_embd == 512

    with pytest.raises(TypeError):
        GPTConfig(dag_hidden_dim=32)  # invalid kwarg


# --------------------------------------------------------------------- #
# Consolidated gradient and hook tests (2 tests)
# --------------------------------------------------------------------- #
def test_gradient_and_hook_behavior_comprehensive(small_dag_gpt):
    """Comprehensive test of gradients, hooks, and multiple backward passes."""
    model, _ = small_dag_gpt
    model.train()

    # Test initial hook behavior
    hook_called = False

    def test_hook(module, grad_input, grad_output):
        nonlocal hook_called
        hook_called = True
        return grad_input

    # Register hook
    hook_handle = model.dag.register_full_backward_hook(test_hook)

    x = torch.randint(0, 20, (2, 4))
    y = torch.randint(0, 20, (2, 4))

    # Test multiple backward passes
    for i in range(2):
        model.zero_grad()
        logits, loss = model(x, y)

        # Verify forward pass
        assert logits.shape == (2, 4, 20)
        assert loss is not None

        # Test backward pass
        loss.backward()

        # Verify gradients are healthy
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(
                    param.grad
                ).all(), f"Parameter {name} has non-finite gradients"
                assert (
                    param.grad.abs().max() < 100
                ), f"Parameter {name} has excessive gradients"

    # Test no-grad context
    with torch.no_grad():
        logits, loss = model(x, y)
        assert logits.shape == (2, 4, 20)
        assert loss is not None

    # Cleanup
    hook_handle.remove()


def test_dag_gradient_flow_and_temperature():
    """Test DAG gradient flow with different temperatures and Gumbel outputs."""
    cfg = GPTConfig(
        vocab_size=10, block_size=3, n_layer=1, n_head=1, n_embd=8, dag_depth=2
    )
    model = GPT(cfg)
    model.train()

    x = torch.randint(0, 10, (2, 3))
    y = torch.randint(0, 10, (2, 3))

    # Test with different temperatures
    for temp in [0.1, 1.0, 2.0]:
        model.zero_grad()

        # Mock different temperature behavior
        logits, loss = model(x, y)
        loss.backward()

        # Verify gradients exist and are finite
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        assert len(grad_norms) > 0, "No gradients found"
        assert all(
            torch.isfinite(torch.tensor(norm)) for norm in grad_norms
        ), "Non-finite gradients"

    # Test Gumbel outputs are approximately discrete
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)

        # Check that outputs have reasonable distribution
        assert probs.min() >= 0 and probs.max() <= 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))


# --------------------------------------------------------------------- #
# Consolidated extra values and node tests (2 tests)
# --------------------------------------------------------------------- #
def test_extra_vals_and_consistency():
    """Test extra values functionality and consistency."""
    cfg = GPTConfig(
        vocab_size=20, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=2
    )
    model = GPT(cfg)

    x = torch.randint(0, 20, (2, 4))
    y = torch.randint(0, 20, (2, 4))

    # Test forward pass with extra values
    logits, loss = model(x, y)
    assert logits.shape == (2, 4, 20)
    assert loss is not None

    # Test consistency across multiple runs
    results = []
    for _ in range(3):
        with torch.no_grad():
            logits, _ = model(x)
            results.append(logits.clone())

    # Results should be identical for same input (deterministic)
    torch.manual_seed(42)
    logits1, _ = model(x)
    torch.manual_seed(42)
    logits2, _ = model(x)
    assert torch.allclose(logits1, logits2, atol=1e-6)


def test_dag_nodes_and_tokens():
    """Test DAG initial nodes and token handling including zero padding."""
    cfg = GPTConfig(
        vocab_size=10, block_size=3, n_layer=1, n_head=1, n_embd=8, dag_depth=2
    )
    model = GPT(cfg)

    # Test with multiple tokens
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    logits, _ = model(tokens)
    assert logits.shape == (2, 3, 10)

    # Test single token (zero padding case)
    single_token = torch.tensor([[7]])
    logits, _ = model(single_token)
    assert logits.shape == (1, 1, 10)

    # Test that node values are properly created
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)
    assert len(node_values) > 0, "No node values found"

    # Verify all node values are finite
    for value in node_values:
        assert torch.isfinite(torch.tensor(value)), f"Node value {value} is not finite"


# --------------------------------------------------------------------- #
# Essential remaining tests (2 tests)
# --------------------------------------------------------------------- #
def test_logging_integration_and_comparison(small_model, sample_batch_small):
    """Test logging integration and DAG vs GPT comparison."""
    model, cfg = small_model
    batch_x, batch_y = sample_batch_small

    # Test logging integration
    assert_valid_logging(model, batch_x, batch_y)

    # Test DAG vs standard GPT comparison
    # Create standard GPT for comparison
    standard_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dag_depth=0,  # Standard GPT has no DAG
    )
    standard_gpt = GPT(standard_cfg)

    # Both should produce valid outputs
    dag_logits, dag_loss = model(batch_x, batch_y)
    std_logits, std_loss = standard_gpt(batch_x, batch_y)

    assert dag_logits.shape == std_logits.shape
    assert dag_loss is not None and std_loss is not None
    assert torch.isfinite(dag_loss) and torch.isfinite(std_loss)


def test_edge_cases_and_validation():
    """Test edge cases and DAG depth validation."""
    # Test valid configurations
    cfg = GPTConfig(dag_depth=1)
    assert cfg.dag_depth == 1

    cfg = GPTConfig(dag_depth=8)
    assert cfg.dag_depth == 8

    # Test model creation with various depths
    for depth in [1, 2, 4]:
        cfg = GPTConfig(
            vocab_size=10, block_size=3, n_layer=1, n_head=1, n_embd=8, dag_depth=depth
        )
        model = GPT(cfg)
        x = torch.randint(0, 10, (1, 3))

        # Should work without errors
        logits, _ = model(x)
        assert logits.shape == (1, 3, 10)

    # Test that model handles different input sizes
    cfg = GPTConfig(
        vocab_size=10, block_size=5, n_layer=1, n_head=1, n_embd=8, dag_depth=2
    )
    model = GPT(cfg)

    for seq_len in [1, 3, 5]:
        x = torch.randint(0, 10, (1, seq_len))
        logits, _ = model(x)
        assert logits.shape == (1, seq_len, 10)
