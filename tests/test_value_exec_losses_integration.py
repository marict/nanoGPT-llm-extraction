"""Integration tests for value_loss and exec_loss in the full training pipeline."""

import types

import pytest
import torch

from data.dagset.streaming import create_dag_structure_dataloaders
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel
from predictor_utils import (compute_dag_structure_loss, evaluate_dag_model,
                             tokenize_texts)


def test_value_exec_losses_with_real_data():
    """Test that value_loss and exec_loss work with real generated DAG data."""

    # Create a simple configuration for testing
    cfg = types.SimpleNamespace(
        max_digits=3,
        max_decimal_places=2,
        sequence_length=128,
        sign_loss_weight=1.0,
        digit_loss_weight=1.0,
        op_loss_weight=1.0,
        value_loss_weight=1.0,
        exec_loss_weight=1.0,
        full_backbone=False,  # Use standalone predictor
    )

    # Create dataloaders with small examples for testing
    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=4,
        val_batch_size=4,
        max_depth=2,  # Small depth for fast testing
        seed=42,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Create a tiny model for testing
    model_config = PredictorOnlyConfig(
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=1,
        n_embd=32,  # Small embedding
        n_head=2,
        dropout=0.0,
        dag_depth=2,
        sequence_length=cfg.sequence_length,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )
    model = PredictorOnlyModel(model_config)

    device = "cpu"
    model.to(device)
    model.eval()

    # Get one batch of validation data
    val_iter = iter(val_loader)
    texts, structures, examples = next(val_iter)

    # Extract targets
    tgt_sgn = structures["initial_sgn"].to(device)
    tgt_digits = structures["initial_digits"].to(device)
    tgt_ops = structures["operation_probs"].to(device)

    # Tokenize inputs
    input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

    # Forward pass
    with torch.no_grad():
        pred_sgn, _, pred_ops = model(input_tokens)
        pred_sgn = pred_sgn.mean(dim=1)
        pred_ops = pred_ops.mean(dim=1)

        # Get digit logits
        last_digit_logits = model.dag_predictor.last_digit_logits.mean(dim=1)

        # Add sequence dimension for loss function compatibility
        pred_sgn = pred_sgn.unsqueeze(1)
        last_digit_logits = last_digit_logits.unsqueeze(1)
        pred_ops = pred_ops.unsqueeze(1)

        # Extract target initial values and final execution values from examples
        batch_size = len(examples)
        target_initial_values = torch.zeros(
            batch_size, 1, tgt_sgn.size(1), device=device
        )
        target_final_exec = torch.zeros(batch_size, 1, device=device)

        for idx, example in enumerate(examples):
            # Target initial values - pad/truncate to match model expectations
            num_values = min(len(example.initial_values), tgt_sgn.size(1))
            target_initial_values[idx, 0, :num_values] = torch.tensor(
                example.initial_values[:num_values], device=device, dtype=torch.float32
            )
            # Target final execution value
            if example.final_value_exec is not None:
                target_final_exec[idx, 0] = torch.tensor(
                    example.final_value_exec, device=device, dtype=torch.float32
                )

        # Compute losses
        losses = compute_dag_structure_loss(
            pred_sgn,
            last_digit_logits,
            pred_ops,
            tgt_sgn.unsqueeze(1),
            tgt_digits.unsqueeze(1),
            tgt_ops.unsqueeze(1),
            target_initial_values,
            target_final_exec,
            cfg,
        )

        # Verify all loss components are present and non-negative
    assert "value_loss" in losses
    assert "exec_loss" in losses
    assert "total_loss" in losses
    assert "sign_loss" in losses
    assert "digit_loss" in losses
    assert "op_loss" in losses

    # All losses should be finite and non-negative
    for loss_name, loss_value in losses.items():
        assert torch.isfinite(loss_value), f"{loss_name} is not finite: {loss_value}"
        assert loss_value.item() >= 0.0, f"{loss_name} is negative: {loss_value}"

    # Total loss should be the sum of weighted individual losses
    expected_total = (
        cfg.sign_loss_weight * losses["sign_loss"]
        + cfg.digit_loss_weight * losses["digit_loss"]
        + cfg.op_loss_weight * losses["op_loss"]
        + cfg.value_loss_weight * losses["value_loss"]
        + cfg.exec_loss_weight * losses["exec_loss"]
    )

    assert pytest.approx(losses["total_loss"].item(), abs=1e-5) == expected_total.item()


def test_evaluation_function_with_new_losses():
    """Test that the evaluate_dag_model function works with the new losses."""

    # Create configuration
    cfg = types.SimpleNamespace(
        max_digits=3,
        max_decimal_places=2,
        sequence_length=64,
        sign_loss_weight=1.0,
        digit_loss_weight=1.0,
        op_loss_weight=1.0,
        value_loss_weight=1.0,
        exec_loss_weight=1.0,
        full_backbone=False,
    )

    # Create dataloaders
    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=2,
        val_batch_size=2,
        max_depth=1,  # Very small for fast testing
        seed=42,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Create model
    model_config = PredictorOnlyConfig(
        vocab_size=50257,
        n_layer=1,
        n_embd=16,  # Very small
        n_head=1,
        dropout=0.0,
        dag_depth=1,
        sequence_length=cfg.sequence_length,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )
    model = PredictorOnlyModel(model_config)

    device = "cpu"
    model.to(device)

    # Run evaluation
    ctx = torch.amp.autocast(device_type="cpu", enabled=False)
    metrics = evaluate_dag_model(
        model=model,
        val_loader=val_loader,
        device=device,
        ctx=ctx,
        cfg=cfg,
        eval_iters=1,  # Just one iteration for testing
        seed=42,
    )

    # Verify that new loss metrics are included
    assert "value_loss" in metrics
    assert "exec_loss" in metrics
    assert "total_loss" in metrics

    # All metrics should be finite
    for metric_name, metric_value in metrics.items():
        assert not torch.isnan(
            torch.tensor(metric_value)
        ), f"{metric_name} is NaN: {metric_value}"
        assert torch.isfinite(
            torch.tensor(metric_value)
        ), f"{metric_name} is not finite: {metric_value}"


def test_zero_weights_exclude_from_total():
    """Test that setting loss weights to zero excludes them from total loss."""

    cfg = types.SimpleNamespace(
        max_digits=3,
        max_decimal_places=2,
        sequence_length=64,
        sign_loss_weight=1.0,
        digit_loss_weight=1.0,
        op_loss_weight=1.0,
        value_loss_weight=0.0,  # Zero weight
        exec_loss_weight=0.0,  # Zero weight
        full_backbone=False,
    )

    # Create small dataset
    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=2,
        val_batch_size=2,
        max_depth=1,
        seed=42,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    model_config = PredictorOnlyConfig(
        vocab_size=50257,
        n_layer=1,
        n_embd=16,
        n_head=1,
        dropout=0.0,
        dag_depth=1,
        sequence_length=cfg.sequence_length,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )
    model = PredictorOnlyModel(model_config)

    device = "cpu"
    model.to(device)

    # Get data and compute losses
    val_iter = iter(val_loader)
    texts, structures, examples = next(val_iter)

    tgt_sgn = structures["initial_sgn"].to(device)
    tgt_digits = structures["initial_digits"].to(device)
    tgt_ops = structures["operation_probs"].to(device)

    input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

    with torch.no_grad():
        pred_sgn, _, pred_ops = model(input_tokens)
        pred_sgn = pred_sgn.mean(dim=1).unsqueeze(1)
        pred_ops = pred_ops.mean(dim=1).unsqueeze(1)
        last_digit_logits = model.dag_predictor.last_digit_logits.mean(dim=1).unsqueeze(
            1
        )

        # Create target tensors
        batch_size = len(examples)
        target_initial_values = torch.zeros(
            batch_size, 1, tgt_sgn.size(1), device=device
        )
        target_final_exec = torch.zeros(batch_size, 1, device=device)

        for idx, example in enumerate(examples):
            num_values = min(len(example.initial_values), tgt_sgn.size(1))
            target_initial_values[idx, 0, :num_values] = torch.tensor(
                example.initial_values[:num_values], device=device, dtype=torch.float32
            )
            if example.final_value_exec is not None:
                target_final_exec[idx, 0] = torch.tensor(
                    example.final_value_exec, device=device, dtype=torch.float32
                )

        losses = compute_dag_structure_loss(
            pred_sgn,
            last_digit_logits,
            pred_ops,
            tgt_sgn.unsqueeze(1),
            tgt_digits.unsqueeze(1),
            tgt_ops.unsqueeze(1),
            target_initial_values,
            target_final_exec,
            cfg,
        )

        # Total loss should only include the non-zero weighted losses
    expected_total = (
        cfg.sign_loss_weight * losses["sign_loss"]
        + cfg.digit_loss_weight * losses["digit_loss"]
        + cfg.op_loss_weight * losses["op_loss"]
        # value_loss and exec_loss should not contribute due to zero weights
    )

    assert pytest.approx(losses["total_loss"].item(), abs=1e-5) == expected_total.item()
