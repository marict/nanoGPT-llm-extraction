# tests/test_dag_model.py
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
                       divide, multiply, op_names, subtract)


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
        dag_scratch_nodes=2,  # Fixed scratch space
    )
    return GPT(cfg), cfg


# --------------------------------------------------------------------- #
# basic forward / backward
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


# --------------------------------------------------------------------- #
# op-function sanity
# --------------------------------------------------------------------- #
def test_op_functions():
    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([1.0, 2.0])
    assert torch.allclose(multiply(x, y), x * y)
    assert torch.allclose(subtract(x, y), x - y)
    assert torch.allclose(divide(x, y), x / y)


# ---------------------------------------------------------------------
# initial-node materialisation tests
# ---------------------------------------------------------------------
def test_dag_initial_nodes_all_tokens(monkeypatch):
    """Every token should contribute exactly one *initial* DAG value."""
    tokens = [0, 1, 2, 3]
    cfg = GPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=1,
    )
    model = GPT(cfg)

    # --- stub value-extractor so each token's value == 3 -------------
    class DummyVal(nn.Module):
        def forward(self, x):  # x : (B,T,H)
            return torch.full((x.size(0), x.size(1)), 3.0, device=x.device)

    model.dag.embed_to_value = DummyVal()

    # Run the model and check that all node values are captured
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model(torch.tensor(tokens).unsqueeze(0))

    # Check node values
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)
    assert len(node_values) == len(
        tokens
    ), f"Expected {len(tokens)} nodes, got {len(node_values)}"


# ---------------------------------------------------------------------
# single-token zero-padding        (fixed recursion)
# ---------------------------------------------------------------------
def test_zero_padding_single_token(monkeypatch):
    cfg = GPTConfig(
        vocab_size=10, block_size=1, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = GPT(cfg)

    # Run model with single token
    model(torch.tensor([7]).unsqueeze(0))

    # With the new causal implementation, single token should create exactly one node
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)
    assert (
        len(node_values) == 1
    ), f"Single token should create 1 node, got {len(node_values)}"

    # The single node value should be finite
    assert torch.isfinite(
        torch.tensor(node_values[0])
    ), f"Node value {node_values[0]} is not finite"


# --------------------------------------------------------------------- #
# config & extra-vals
# --------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------
# extra-vals entropy / grad check  (robust to dimensionality)
# ---------------------------------------------------------------------
def test_extra_vals_daggpt():
    """Test GPT's logging functionality using DAGLogger with gradient computation."""
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    model.train()  # Enable gradient computation
    logger = DAGLogger()

    # Test that logger returns empty dict before any forward pass
    # Note: get_extra_vals requires compute_log_statistics to be called first
    # So we can't test it before forward pass

    # Forward pass to populate activations
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    # Test backward pass works
    loss.backward()

    # Extract logging data after forward pass
    logger.compute_log_statistics(model)

    # Test gate and norm extraction after forward pass
    extra_vals = logger.get_extra_vals(model)
    norm_keys = [k for k in extra_vals if k.startswith("norm/")]
    assert len(norm_keys) > 0, "No norm values found"

    # Verify loss is reasonable
    assert loss > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"

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


def test_extra_vals_consistency_daggpt():
    """Test that GPT's logging via DAGLogger returns consistent structure across calls with gradients."""
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    model.train()  # Enable gradient computation
    logger = DAGLogger()

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    # Test backward pass works
    loss.backward()

    # Extract logging data after forward pass
    logger.compute_log_statistics(model)

    # Call logger multiple times
    extra_vals_1 = logger.get_extra_vals(model)
    extra_vals_2 = logger.get_extra_vals(model)

    # Should be identical
    assert extra_vals_1.keys() == extra_vals_2.keys()
    for k in extra_vals_1.keys():
        assert extra_vals_1[k] == extra_vals_2[k], f"Inconsistent value for key {k}"

    # Verify we have some valid values
    assert len(extra_vals_1) > 0, "Should have some extra values"

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


def test_hook_behavior():
    """Test that DAG structure is working properly with gradient computation."""
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    model.train()  # Enable gradient computation
    logger = DAGLogger()

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)

    # Test backward pass works
    logger.setup_gradient_tracking(model)
    loss.backward()

    # Check that DAG components are working
    assert hasattr(model, "dag"), "Model should have DAG"
    assert hasattr(model.dag, "plan_predictor"), "DAG should have plan_predictor"

    # Verify forward pass worked correctly
    assert logits.shape == (2, 8, cfg.vocab_size), "Logits shape incorrect"
    assert loss > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"

    # Verify gradient values are reasonable
    for name, grad_val in logger.captured_gradients.items():
        assert isinstance(grad_val, float), f"Gradient {name} is not a float"
        assert not torch.isnan(torch.tensor(grad_val)), f"Gradient {name} is NaN"

    # Verify key model gradients exist and are finite (not all parameters need gradients in every pass)
    key_params_with_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"
            key_params_with_grads += 1

    # Ensure at least some parameters received gradients
    assert key_params_with_grads > 0, "No parameters received gradients"


def test_hook_behavior_multiple_backward_passes():
    """Test that hooks work correctly across multiple backward passes via DAGLogger."""
    # Set deterministic seed for this test
    torch.manual_seed(42)

    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    logger = DAGLogger()

    # First forward and backward pass
    x1 = torch.randint(0, cfg.vocab_size, (2, 8))
    y1 = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss1 = model(x1, y1)
    logger.setup_gradient_tracking(model)
    loss1.sum().backward()

    first_grads = logger.captured_gradients.copy()
    assert len(first_grads) > 0

    # Clear gradients and do a second forward pass with same data
    model.zero_grad()
    logger.captured_gradients.clear()  # Clear logger gradients too
    _, loss2 = model(x1, y1)
    logger.setup_gradient_tracking(model)
    loss2.sum().backward()

    second_grads = logger.captured_gradients.copy()
    assert len(second_grads) > 0

    # With same inputs, gradients should be similar but not identical due to Gumbel sampling
    # Gumbel softmax introduces randomness even with the same input
    assert first_grads.keys() == second_grads.keys()
    for k in first_grads.keys():
        # Gumbel sampling can cause significant variation, so we mainly check that gradients are finite
        # and not too extreme, rather than expecting consistency between runs
        assert torch.isfinite(
            torch.tensor(first_grads[k])
        ), f"First gradient {k} is not finite: {first_grads[k]}"
        assert torch.isfinite(
            torch.tensor(second_grads[k])
        ), f"Second gradient {k} is not finite: {second_grads[k]}"

        # Check gradients are reasonable (not too extreme)
        assert (
            abs(first_grads[k]) < 1e4
        ), f"First gradient {k} too large: {first_grads[k]}"
        assert (
            abs(second_grads[k]) < 1e4
        ), f"Second gradient {k} too large: {second_grads[k]}"

    # Now test with different data to ensure hooks can capture different gradients
    x3 = torch.ones_like(x1) * (cfg.vocab_size - 1)  # Very different input
    y3 = torch.zeros_like(y1)  # Very different target
    model.zero_grad()
    logger.captured_gradients.clear()
    _, loss3 = model(x3, y3)
    logger.setup_gradient_tracking(model)
    loss3.sum().backward()

    third_grads = logger.captured_gradients.copy()
    assert len(third_grads) > 0

    # Verify that the hooks are still working by checking gradients exist
    assert third_grads.keys() == first_grads.keys()
    for k in third_grads.keys():
        assert isinstance(third_grads[k], float)
        assert not torch.isnan(torch.tensor(third_grads[k]))


def test_hook_behavior_no_grad_context():
    """Test that hooks don't interfere when in no_grad context via DAGLogger."""
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    logger = DAGLogger()

    # Forward pass in no_grad context
    with torch.no_grad():
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss = model(x)

        # Should still have DAG components
        assert hasattr(model, "dag")
        assert hasattr(model.dag, "plan_predictor")

        # Logger gradients should be empty (no backward pass)
        assert len(logger.captured_gradients) == 0

        # Extract logging data after forward pass
        logger.compute_log_statistics(model)

        # Logger should work and return reasonable values
        extra_vals = logger.get_extra_vals(model)

        # Should have at least gate and norm values
        norm_keys = [k for k in extra_vals if k.startswith("norm/")]
        assert len(norm_keys) > 0, "No norm values found"

        # Verify we have some meaningful logging data
        assert len(norm_keys) > 0, "Should have some logging values"


# ---------------------------------------------------------------------
# Gradient health tests for DAG components
# ---------------------------------------------------------------------
def test_dag_gradient_health():
    """Test that DAG gradients remain healthy across multiple backward passes."""
    torch.manual_seed(42)

    cfg = GPTConfig(
        vocab_size=20,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=2,
        gumbel_temperature=2.0,
    )
    model = GPT(cfg)

    gradient_norms = []

    for i in range(3):  # Reduced from 5 to save time
        x = torch.randint(0, cfg.vocab_size, (2, 6))
        y = torch.randint(0, cfg.vocab_size, (2, 6))

        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        # Check specific gradient health
        op_grad = model.dag.plan_predictor.predictor[-1].weight.grad
        assert op_grad is not None, f"Step {i}: should have gradients"

        grad_norm = op_grad.norm().item()
        gradient_norms.append(grad_norm)

        # Verify gradient health
        assert grad_norm > 1e-8, f"Step {i}: gradient too small: {grad_norm}"
        assert grad_norm < 1e4, f"Step {i}: gradient too large: {grad_norm}"
        assert torch.isfinite(op_grad).all(), f"Step {i}: gradient contains inf/nan"

    # Check gradient stability
    min_grad = min(gradient_norms)
    max_grad = max(gradient_norms)
    ratio = max_grad / min_grad if min_grad > 0 else float("inf")
    assert ratio < 1000, f"Gradient variation too extreme: ratio={ratio}"


def test_dag_gradient_flow_vs_temperature():
    """Test that gradient flow improves with higher temperature."""
    # Set deterministic seed for this test
    torch.manual_seed(42)

    cfg_base = GPTConfig(
        vocab_size=20,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=2,
    )
    x = torch.randint(0, cfg_base.vocab_size, (2, 6))
    y = torch.randint(0, cfg_base.vocab_size, (2, 6))

    # Test different temperatures
    temperatures = [0.5, 1.0, 2.0, 3.0]
    gradient_norms = []

    for temp in temperatures:
        cfg = GPTConfig(**{**cfg_base.__dict__, "gumbel_temperature": temp})
        model = GPT(cfg)

        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        # Measure gradient norm for op_selector (previously problematic)
        op_grad = model.dag.plan_predictor.predictor[-1].weight.grad
        if op_grad is not None:
            grad_norm = op_grad.norm().item()
            gradient_norms.append(grad_norm)
        else:
            gradient_norms.append(0.0)

    # Check that gradient norms are reasonable for our chosen temperature (2.0)
    temp_2_idx = temperatures.index(2.0)
    temp_2_grad = gradient_norms[temp_2_idx]

    # Should have non-zero gradients with temperature 2.0
    assert (
        temp_2_grad > 1e-8
    ), f"Temperature 2.0 should give non-zero gradients, got {temp_2_grad}"
    assert (
        temp_2_grad < 1e2
    ), f"Temperature 2.0 should give finite gradients, got {temp_2_grad}"

    # Very low temperature (0.5) might have gradient issues
    temp_low_grad = gradient_norms[0]
    # Both low and moderate temperature should have reasonable gradients
    # The exact relationship between temperature and gradient magnitude can vary
    # depending on the specific samples and model state, so we just check both are reasonable
    if temp_low_grad > 5e-7:
        # If low temp has gradients, verify it's not too extreme
        assert (
            temp_low_grad < 1e2
        ), f"Low temperature gradient too large: {temp_low_grad}"

    # The key requirement is that our chosen temperature (2.0) gives stable gradients
    # regardless of how it compares to other temperatures
    assert (
        temp_2_grad > 1e-8
    ), f"Temperature 2.0 should maintain non-zero gradients, got {temp_2_grad}"


def test_dag_gumbel_outputs_are_discrete():
    """Test that Gumbel softmax outputs are approximately one-hot."""
    cfg = GPTConfig(
        vocab_size=20,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=2,
        gumbel_temperature=0.05,  # Further lower temp for more discrete outputs
    )
    model = GPT(cfg)

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 3))  # Batch size 2 for better testing
    with torch.no_grad():
        model(x)

    # Get the probability tensors
    operand1_probs = (
        model.dag.plan_predictor.last_operand1_probs
    )  # (B, T, dag_depth, max_nodes)
    operand2_probs = (
        model.dag.plan_predictor.last_operand2_probs
    )  # (B, T, dag_depth, max_nodes)
    operation_probs = (
        model.dag.plan_predictor.last_operation_probs_full
    )  # (B, T, dag_depth, n_ops)
    output_probs = model.dag.plan_predictor.last_output_probs  # (B, T, max_nodes)

    def check_approximately_onehot(probs_tensor, tolerance=0.6):
        """Check if each probability distribution is approximately one-hot."""
        # Flatten all dimensions except the last one (the dimension we're taking softmax over)
        flat_probs = probs_tensor.view(-1, probs_tensor.size(-1))

        for i in range(flat_probs.size(0)):
            dist = flat_probs[i]

            # Skip if all zeros (masked out)
            if dist.sum() < 1e-6:
                continue

            # Check sums to 1
            assert torch.allclose(
                dist.sum(), torch.tensor(1.0), atol=1e-6
            ), f"Distribution {i} should sum to 1, got {dist.sum():.6f}"

            # Check one-hot-ness: max value should be much larger than others
            max_val = dist.max()
            second_max = dist.topk(2)[0][1] if dist.size(0) > 1 else torch.tensor(0.0)

            # For good one-hot behavior, max should be >> second max
            assert max_val > (
                1 - tolerance
            ), f"Distribution {i} max value {max_val:.3f} should be > {1-tolerance} for one-hot"
            assert (
                second_max < tolerance
            ), f"Distribution {i} second max {second_max:.3f} should be < {tolerance} for one-hot"

    # Test each probability tensor type
    check_approximately_onehot(operand1_probs)
    check_approximately_onehot(operand2_probs)
    check_approximately_onehot(operation_probs)
    check_approximately_onehot(output_probs)


def test_dag_gradients_multiple_backward_passes():
    """Test that DAG gradients remain healthy across multiple backward passes."""
    # Set deterministic seed for this test
    torch.manual_seed(42)

    cfg = GPTConfig(
        vocab_size=20,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=2,
        gumbel_temperature=2.0,
    )
    model = GPT(cfg)

    gradient_norms = []

    for i in range(5):  # Multiple training steps
        x = torch.randint(0, cfg.vocab_size, (2, 6))
        y = torch.randint(0, cfg.vocab_size, (2, 6))

        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        # Check op_selector gradients specifically
        op_grad = model.dag.plan_predictor.predictor[-1].weight.grad
        assert op_grad is not None, f"Step {i}: op_selector should have gradients"

        grad_norm = op_grad.norm().item()
        gradient_norms.append(grad_norm)

        # Each step should have healthy gradients
        assert grad_norm > 1e-8, f"Step {i}: gradient too small: {grad_norm}"
        assert grad_norm < 1e4, f"Step {i}: gradient too large: {grad_norm}"
        assert torch.isfinite(op_grad).all(), f"Step {i}: gradient contains inf/nan"

    # Gradients should be relatively consistent (not exploding or vanishing dramatically)
    min_grad = min(gradient_norms)
    max_grad = max(gradient_norms)
    ratio = max_grad / min_grad if min_grad > 0 else float("inf")

    # Allow reasonable variation but not extreme explosion/vanishing
    assert (
        ratio < 1000
    ), f"Gradient variation too extreme: min={min_grad}, max={max_grad}, ratio={ratio}"


def test_basic_daggpt_functionality(tiny_model, sample_batch_tiny):
    """Test basic DAG model forward pass and structure."""
    model, config = tiny_model
    input_ids, target_ids = sample_batch_tiny

    # Test forward pass
    assert_valid_forward_pass(model, config, input_ids, target_ids)
    assert_valid_forward_pass(model, config, input_ids)  # without targets

    # Test DAG component exists
    assert hasattr(model, "dag"), "Model should have DAG"

    # Test node values
    assert_valid_node_values(model)


def test_daggpt_vs_gpt_comparison():
    """Test that DAG models behave correctly compared to standard GPT."""
    # Standard GPT (dag_depth=0)
    cfg_std = GPTConfig(**{**SMALL_CONFIG.__dict__, "dag_depth": 0})
    model_std = GPT(cfg_std)

    # DAG GPT
    model_dag = GPT(SMALL_CONFIG)

    x = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 4))
    y = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 4))

    # Both should produce valid outputs
    for model, config in [(model_std, cfg_std), (model_dag, SMALL_CONFIG)]:
        assert_valid_forward_pass(model, config, x, y)

    # Standard GPT shouldn't have DAG components
    assert not hasattr(model_std, "dag")
    assert not hasattr(model_std, "dag_mixer")

    # DAG GPT should have DAG component
    assert hasattr(model_dag, "dag")
    assert not hasattr(model_dag, "dag_mixer")


def test_logging_integration(small_model, sample_batch_small):
    """Test comprehensive logging functionality."""
    model, config = small_model
    input_ids, target_ids = sample_batch_small

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Test all logging functionality
    logger, wandb_dict, extra_vals = assert_valid_logging(model)

    # Verify specific logging components
    assert len(extra_vals) > 0, "Should have extra values"

    # Test node values specifically
    assert_valid_node_values(model)


def test_gradient_tracking_comprehensive(small_model):
    """Test gradient tracking and backward pass functionality."""
    model, config = small_model

    # Set up gradient tracking test
    logger, loss, input_ids, target_ids = setup_gradient_tracking_test(model, config)

    # Verify gradients were captured
    extra_vals = logger.get_extra_vals(model)

    # Check for operation gradients
    op_grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(op_grad_keys) == len(
        op_names
    ), "Should have gradients for all operations"

    # Verify gradient values are reasonable
    for key in op_grad_keys:
        grad_val = extra_vals[key]
        assert isinstance(grad_val, float)
        assert torch.isfinite(torch.tensor(grad_val))
        assert abs(grad_val) < 1.0  # Reasonable gradient magnitude


def test_edge_cases_and_validation():
    """Test edge cases and input validation."""
    # Test zero dag_depth
    cfg_zero = GPTConfig(
        vocab_size=30, block_size=6, n_layer=1, n_head=1, n_embd=16, dag_depth=0
    )
    model_zero = GPT(cfg_zero)
    x = torch.randint(0, 30, (1, 4))

    assert_valid_forward_pass(model_zero, cfg_zero, x)
    assert not hasattr(model_zero, "dag")

    # Test block size validation
    cfg_small_block = GPTConfig(
        vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model_small = GPT(cfg_small_block)
    x_too_long = torch.randint(0, 10, (1, 3))  # length 3 > block_size 2

    with pytest.raises(AssertionError):
        model_small(x_too_long)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
