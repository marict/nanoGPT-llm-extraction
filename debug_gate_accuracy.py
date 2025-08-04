#!/usr/bin/env python3
"""
Debug gate accuracy issues - investigate why it's stuck at 40%
"""

import torch
import torch.nn.functional as F

from data.dagset.streaming import create_dag_structure_dataloaders
from predictor_utils import _compute_g_loss


def debug_gate_targets_and_masking():
    """Debug gate target generation and masking logic."""
    print("=== DEBUGGING GATE ACCURACY ISSUES ===\n")

    # Create a sample batch
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=4,
        val_batch_size=4,
        max_depth=4,
        seed=42,
        max_digits=4,
        max_decimal_places=4,
        block_size=32,
    )

    texts, target_tensors, valid_mask = next(train_loader)

    # Extract a few valid positions to analyze
    valid_positions = valid_mask.nonzero(as_tuple=False)[
        :10
    ]  # First 10 valid positions

    print(f"Analyzing {len(valid_positions)} valid positions:\n")

    # Extract targets for valid positions
    target_G_list = []
    target_O_list = []

    for i, (batch_idx, token_idx) in enumerate(valid_positions):
        target_dict = target_tensors[batch_idx.item()][token_idx.item()]
        target_G = target_dict["target_G"]  # (dag_depth,)
        target_O = target_dict["target_O"]  # (dag_depth, total_nodes)

        target_G_list.append(target_G)
        target_O_list.append(target_O)

        print(f"Position {i}:")
        print(f"  Text: '{texts[batch_idx.item()]}'")
        print(f"  Target G: {target_G.tolist()}")

        # Check the masking logic
        O_step_sums = torch.sum(torch.abs(target_O), dim=-1)  # (dag_depth,)
        operation_mask = ~torch.isclose(
            O_step_sums, torch.ones_like(O_step_sums), atol=1e-6
        )

        print(f"  O step sums: {O_step_sums.tolist()}")
        print(f"  Operation mask: {operation_mask.tolist()}")
        print(f"  Masked G targets: {target_G[operation_mask].tolist()}")
        print()

    # Stack all targets
    if target_G_list:
        stacked_target_G = torch.stack(target_G_list)  # (num_valid, dag_depth)
        stacked_target_O = torch.stack(
            target_O_list
        )  # (num_valid, dag_depth, total_nodes)

        print("=== OVERALL STATISTICS ===")
        print(f"Target G shape: {stacked_target_G.shape}")
        print(f"Target O shape: {stacked_target_O.shape}")

        # Global masking
        O_step_sums_global = torch.sum(
            torch.abs(stacked_target_O), dim=-1
        )  # (num_valid, dag_depth)
        operation_mask_global = ~torch.isclose(
            O_step_sums_global, torch.ones_like(O_step_sums_global), atol=1e-6
        )

        print(f"Global operation mask shape: {operation_mask_global.shape}")
        print(f"Total gate positions: {operation_mask_global.numel()}")
        print(f"Actual operations: {operation_mask_global.sum().item()}")
        print(f"Identity operations: {(~operation_mask_global).sum().item()}")

        if operation_mask_global.any():
            masked_targets = stacked_target_G[operation_mask_global]
            print(
                f"Masked target distribution: {torch.bincount(masked_targets.long()).tolist()}"
            )
            print(f"Average masked target: {masked_targets.mean().item():.3f}")

        # Test with perfect predictions (all 1s)
        pred_G_perfect_1s = torch.ones_like(stacked_target_G)  # All predict 1
        pred_G_sigmoid_1s = torch.sigmoid(pred_G_perfect_1s * 10)  # Strong sigmoid to 1

        _, acc_perfect_1s = _compute_g_loss(
            pred_G_perfect_1s * 10, stacked_target_G, stacked_target_O
        )
        print(f"Accuracy if always predicting 1: {acc_perfect_1s.item():.3f}")

        # Test with perfect predictions (all 0s)
        pred_G_perfect_0s = torch.zeros_like(stacked_target_G)  # All predict 0
        pred_G_sigmoid_0s = torch.sigmoid(pred_G_perfect_0s - 10)  # Strong sigmoid to 0

        _, acc_perfect_0s = _compute_g_loss(
            pred_G_perfect_0s - 10, stacked_target_G, stacked_target_O
        )
        print(f"Accuracy if always predicting 0: {acc_perfect_0s.item():.3f}")

        # Test with random predictions
        pred_G_random = torch.randn_like(stacked_target_G)
        _, acc_random = _compute_g_loss(
            pred_G_random, stacked_target_G, stacked_target_O
        )
        print(f"Accuracy with random predictions: {acc_random.item():.3f}")

        print("\n=== ISSUE DIAGNOSIS ===")
        if acc_perfect_1s.item() < 0.6:
            print("🚨 ISSUE: Even perfect '1' predictions get low accuracy!")
            print("   This suggests target generation or masking issues.")

        if acc_random.item() > 0.3:
            print("🚨 ISSUE: Random predictions get suspiciously high accuracy!")
            print("   This suggests the task is too easy or targets are biased.")

        if operation_mask_global.sum().item() < 5:
            print("🚨 ISSUE: Very few real operations found!")
            print("   This suggests masking is too aggressive.")


if __name__ == "__main__":
    debug_gate_targets_and_masking()
