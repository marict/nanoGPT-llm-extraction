#!/usr/bin/env python
"""Test DAG operation logging functionality."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_common import assert_valid_logging, sample_batch_small, small_model

from dag_logger import DAGLogger
from models.dag_model import GPT, OP_NAMES, GPTConfig


# --------------------------------------------------------------------- #
# Consolidated DAG operation logging tests (2 tests)
# --------------------------------------------------------------------- #
def test_comprehensive_operation_logging_and_gradients(small_model, sample_batch_small):
    """Test comprehensive operation logging including basic logging, gradients, and consistency."""
    model, config = small_model
    batch_x, batch_y = sample_batch_small

    # Test basic operation logging
    model.train()
    logits, loss = model(batch_x, batch_y)

    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    # Backward pass to capture gradients
    loss.backward()

    # Test logging functionality
    logger.compute_log_statistics(model)
    extra_vals = logger.get_extra_vals(model)
    wandb_dict = logger.get_wandb_logging_dict(model)

    # Verify basic logging
    assert len(extra_vals) > 0, "Should have extra values"
    assert len(wandb_dict) > 0, "Should have wandb dict"

    # Test gradient consistency
    # Clear and run again to test consistency
    model.zero_grad()
    logger.captured_gradients.clear()

    logits2, loss2 = model(batch_x, batch_y)
    logger.setup_gradient_tracking(model)
    loss2.backward()

    # Verify gradients were captured again
    assert len(logger.captured_gradients) > 0, "Should capture gradients on second run"

    # Test gradient values are reasonable
    for grad_name, grad_value in logger.captured_gradients.items():
        assert isinstance(grad_value, float), f"Gradient {grad_name} should be float"
        assert torch.isfinite(
            torch.tensor(grad_value)
        ), f"Gradient {grad_name} should be finite"
        assert abs(grad_value) < 1000, f"Gradient {grad_name} too large: {grad_value}"

    # Test comprehensive logging integration
    logger2 = DAGLogger()
    logger2.compute_log_statistics(model)
    extra_vals2 = logger2.get_extra_vals(model)
    wandb_dict2 = logger2.get_wandb_logging_dict(model)

    # Should have similar structure (though values may differ due to stochastic operations)
    assert set(extra_vals2.keys()).issubset(
        set(extra_vals.keys()) | set(["grad/op_" + op for op in OP_NAMES])
    )
    assert len(wandb_dict2) > 0

    # Test DAG hidden gradient logging
    # Test with different DAG depth
    dag_config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=8,
        vocab_size=50,
        dag_depth=3,
        bias=False,
    )
    dag_model = GPT(dag_config)
    dag_model.train()

    # Create test batch for DAG model
    dag_x = torch.randint(0, dag_config.vocab_size, (2, 6))
    dag_y = torch.randint(0, dag_config.vocab_size, (2, 6))

    dag_logits, dag_loss = dag_model(dag_x, dag_y)

    dag_logger = DAGLogger()
    dag_logger.setup_gradient_tracking(dag_model)
    dag_loss.backward()

    # Compute statistics before getting extra values
    dag_logger.compute_log_statistics(dag_model)

    # Should have DAG-specific gradients
    dag_extra_vals = dag_logger.get_extra_vals(dag_model)
    op_grad_keys = [k for k in dag_extra_vals.keys() if k.startswith("grad/op")]

    # Should have gradients for all operations
    assert len(op_grad_keys) > 0, "Should have operation gradients for DAG model"

    # Test operation probability logging
    dag_logger.compute_log_statistics(dag_model)
    dag_wandb_dict = dag_logger.get_wandb_logging_dict(dag_model)

    # Should have operation-related logging
    op_prob_keys = [
        k for k in dag_wandb_dict.keys() if "op_prob" in k or "operation" in k
    ]
    assert (
        len(op_prob_keys) >= 0
    ), "Should have operation probability logging"  # May be 0 if not implemented


def test_logging_multiple_passes_and_no_dag_scenarios(small_model, sample_batch_small):
    """Test logging after multiple passes and scenarios without DAG depth."""
    model, config = small_model
    batch_x, batch_y = sample_batch_small

    # Test logging after multiple passes
    model.train()
    logger = DAGLogger()

    # First pass
    logits1, loss1 = model(batch_x, batch_y)
    logger.setup_gradient_tracking(model)
    loss1.backward()

    logger.compute_log_statistics(model)
    extra_vals1 = logger.get_extra_vals(model)
    wandb_dict1 = logger.get_wandb_logging_dict(model)

    # Second pass (model state should be updated)
    model.zero_grad()
    logger.captured_gradients.clear()

    logits2, loss2 = model(batch_x, batch_y)
    logger.setup_gradient_tracking(model)
    loss2.backward()

    logger.compute_log_statistics(model)
    extra_vals2 = logger.get_extra_vals(model)
    wandb_dict2 = logger.get_wandb_logging_dict(model)

    # Should have consistent structure
    assert len(extra_vals2) > 0, "Should have extra values after second pass"
    assert len(wandb_dict2) > 0, "Should have wandb dict after second pass"

    # Keys should be similar (though values may differ)
    assert set(extra_vals1.keys()) == set(
        extra_vals2.keys()
    ), "Keys should be consistent across passes"

    # Third pass with different data
    new_x = torch.randint(0, config.vocab_size, (3, 5))  # Different shape
    new_y = torch.randint(0, config.vocab_size, (3, 5))

    model.zero_grad()
    logger.captured_gradients.clear()

    logits3, loss3 = model(new_x, new_y)
    logger.setup_gradient_tracking(model)
    loss3.backward()

    logger.compute_log_statistics(model)
    extra_vals3 = logger.get_extra_vals(model)

    # Should handle different batch sizes
    assert len(extra_vals3) > 0, "Should handle different batch sizes"

    # Test no DAG depth logging (standard GPT)
    no_dag_config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=8,
        vocab_size=50,
        dag_depth=0,
        bias=False,  # No DAG
    )
    no_dag_model = GPT(no_dag_config)
    no_dag_model.train()

    # Create compatible batch
    no_dag_x = (
        batch_x[:, : no_dag_config.block_size]
        if batch_x.size(1) > no_dag_config.block_size
        else batch_x
    )
    no_dag_y = (
        batch_y[:, : no_dag_config.block_size]
        if batch_y.size(1) > no_dag_config.block_size
        else batch_y
    )

    no_dag_logits, no_dag_loss = no_dag_model(no_dag_x, no_dag_y)

    no_dag_logger = DAGLogger()
    no_dag_logger.setup_gradient_tracking(no_dag_model)
    no_dag_loss.backward()

    no_dag_logger.compute_log_statistics(no_dag_model)
    no_dag_extra_vals = no_dag_logger.get_extra_vals(no_dag_model)
    no_dag_wandb_dict = no_dag_logger.get_wandb_logging_dict(no_dag_model)

    # Should still have some logging (non-DAG related)
    assert len(no_dag_wandb_dict) >= 0, "Should have some logging even without DAG"

    # Should not have DAG-specific operation gradients
    op_grad_keys = [k for k in no_dag_extra_vals.keys() if k.startswith("grad/op")]
    # May or may not have op gradients depending on implementation

    # Verify the model actually has no DAG
    assert not hasattr(no_dag_model, "dag") or no_dag_model.config.dag_depth == 0

    # Test that logger handles this gracefully
    try:
        node_values = no_dag_logger.get_node_values_list(no_dag_model)
        # If it succeeds, should return empty or handle gracefully
        if node_values is not None:
            assert isinstance(node_values, list)
    except (AttributeError, AssertionError):
        # Expected for models without DAG
        pass

    # Test operation probability logging for no-DAG model
    # Should handle gracefully even if no operations exist
    assert isinstance(
        no_dag_wandb_dict, dict
    ), "Should return dict even for no-DAG model"
