#!/usr/bin/env python3
"""
Benchmark training components to identify performance bottlenecks.
"""

import sys
import time
from contextlib import contextmanager

import torch

sys.path.append(".")

from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import GPT, GPTConfig
from predictor_utils import compute_dag_loss, tokenize_texts


@contextmanager
def timer(name):
    """Context manager to time operations."""
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {(end - start) * 1000:.2f}ms")


def benchmark_data_generation():
    """Benchmark data generation pipeline."""
    print("\n=== Benchmarking Data Generation ===")

    # Create data loaders
    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=4,  # Small batch for testing
        val_batch_size=4,
        max_depth=4,
        seed=42,
        max_digits=4,
        max_decimal_places=4,
        block_size=64,
    )

    # Time data loading
    with timer("Data generation (1 batch)"):
        texts, target_tensors, valid_mask = next(train_loader)

    print(f"Batch size: {len(texts)}")
    print(f"Sequence length: {len(target_tensors[0])}")
    print(f"Valid positions: {valid_mask.sum().item()} / {valid_mask.numel()}")

    # Time multiple batches
    total_time = 0
    num_batches = 10
    for i in range(num_batches):
        start = time.time()
        texts, target_tensors, valid_mask = next(train_loader)
        total_time += time.time() - start

    avg_time = (total_time / num_batches) * 1000
    print(f"Average data generation time ({num_batches} batches): {avg_time:.2f}ms")

    return texts, target_tensors, valid_mask


def benchmark_model_forward():
    """Benchmark model forward pass."""
    print("\n=== Benchmarking Model Forward Pass ===")

    # Create model config
    model_cfg_dict = {
        "vocab_size": 50304,
        "n_embd": 256,  # Smaller for testing
        "n_head": 8,
        "n_layer": 6,  # Smaller for testing
        "dropout": 0.0,
        "bias": True,
        "dag_depth": 4,
        "block_size": 64,
        "max_digits": 4,
        "max_decimal_places": 4,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    with timer("Model creation"):
        model_config = GPTConfig(**model_cfg_dict)
        model = GPT(model_config)
        model = model.to(device)
        model.eval()

    # Get sample data
    texts, target_tensors, valid_mask = benchmark_data_generation()

    # Time tokenization
    with timer("Tokenization"):
        input_tokens = tokenize_texts(texts, model_cfg_dict["block_size"], device)

    # Time forward pass
    with torch.no_grad():
        with timer("Model forward (hidden states)"):
            hidden_states = model.forward_hidden(input_tokens)

        with timer("DAG predictor forward"):
            pred_digit_logits, pred_V_sign, pred_O, pred_G = model.dag_predictor(
                hidden_states
            )

    # Create a mock cfg for loss computation
    class MockCfg:
        def __init__(self):
            self.max_digits = 4
            self.max_decimal_places = 4
            self.enable_exec_loss = True
            self.exec_loss_weight = 0.01

    cfg = MockCfg()
    return (
        model,
        texts,
        target_tensors,
        valid_mask,
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        cfg,
        device,
    )


def benchmark_loss_computation():
    """Benchmark loss computation components."""
    print("\n=== Benchmarking Loss Computation ===")

    (
        model,
        texts,
        target_tensors,
        valid_mask,
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        cfg,
        device,
    ) = benchmark_model_forward()

    # Move data to device
    valid_mask = valid_mask.to(device)
    for seq_targets in target_tensors:
        for target_dict in seq_targets:
            for key, tensor in target_dict.items():
                if isinstance(tensor, torch.Tensor):
                    target_dict[key] = tensor.to(device)

    # Time full loss computation
    with timer("Full DAG loss computation"):
        dag_executor = getattr(model, "dag_executor", None)
        losses = compute_dag_loss(
            pred_digit_logits,
            pred_V_sign,
            pred_O,
            pred_G,
            target_tensors,
            valid_mask,
            dag_executor=dag_executor,
            cfg=cfg,
        )

    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Exec loss: {losses.get('exec_loss', 0.0):.4f}")

    # Test without exec loss
    cfg.enable_exec_loss = False
    with timer("DAG loss without exec_loss"):
        losses_no_exec = compute_dag_loss(
            pred_digit_logits,
            pred_V_sign,
            pred_O,
            pred_G,
            target_tensors,
            valid_mask,
            dag_executor=None,  # No executor
            cfg=cfg,
        )

    print(f"Total loss (no exec): {losses_no_exec['total_loss'].item():.4f}")


def benchmark_training_iteration():
    """Benchmark a full training iteration."""
    print("\n=== Benchmarking Full Training Iteration ===")

    model, texts, target_tensors, valid_mask, _, _, _, _, cfg, device = (
        benchmark_model_forward()
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Move data to device
    valid_mask = valid_mask.to(device)
    for seq_targets in target_tensors:
        for target_dict in seq_targets:
            for key, tensor in target_dict.items():
                if isinstance(tensor, torch.Tensor):
                    target_dict[key] = tensor.to(device)

    # Time full iteration
    with timer("Full training iteration"):
        optimizer.zero_grad()

        # Forward pass
        input_tokens = tokenize_texts(texts, 64, device)
        hidden_states = model.forward_hidden(input_tokens)
        pred_digit_logits, pred_V_sign, pred_O, pred_G = model.dag_predictor(
            hidden_states
        )

        # Loss computation
        dag_executor = getattr(model, "dag_executor", None)
        losses = compute_dag_loss(
            pred_digit_logits,
            pred_V_sign,
            pred_O,
            pred_G,
            target_tensors,
            valid_mask,
            dag_executor=dag_executor,
            cfg=cfg,
        )

        loss = losses["total_loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("🚀 Starting Training Performance Benchmark")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    benchmark_data_generation()
    benchmark_loss_computation()
    benchmark_training_iteration()

    print("\n✅ Benchmark completed!")
