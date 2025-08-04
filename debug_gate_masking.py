#!/usr/bin/env python3
"""
Debug the gate masking logic - is it too aggressive?
"""

import torch

from data.dagset.streaming import create_dag_structure_dataloaders, tensor_to_expression


def debug_masking_logic():
    """Debug whether the masking logic is correctly identifying real operations."""
    print("=== DEBUGGING GATE MASKING LOGIC ===\n")

    # Create sample data
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=2,
        val_batch_size=2,
        max_depth=4,
        seed=42,
        max_digits=4,
        max_decimal_places=4,
        block_size=32,
    )

    texts, target_tensors, valid_mask = next(train_loader)

    # Look at a few valid positions
    valid_positions = valid_mask.nonzero(as_tuple=False)[:5]

    for i, (batch_idx, token_idx) in enumerate(valid_positions):
        target_dict = target_tensors[batch_idx.item()][token_idx.item()]

        text = texts[batch_idx.item()]
        target_digits = target_dict["target_digits"]
        target_V_sign = target_dict["target_V_sign"]
        target_O = target_dict["target_O"]  # (dag_depth, total_nodes)
        target_G = target_dict["target_G"]  # (dag_depth,)

        print(f"=== Position {i} ===")
        print(f"Text: '{text}'")

        # Reconstruct the target expression
        try:
            target_expr = tensor_to_expression(
                target_digits,
                target_V_sign,
                target_O,
                target_G,
                max_digits=4,
                max_decimal_places=4,
            )
            print(f"Target expression: {target_expr}")
        except Exception as e:
            print(f"Could not reconstruct expression: {e}")

        print(f"Target G: {target_G.tolist()}")

        # Analyze each operation step
        for step in range(target_O.shape[0]):  # dag_depth
            O_step = target_O[step]  # (total_nodes,)
            O_step_sum = torch.sum(torch.abs(O_step)).item()
            G_step = target_G[step].item()

            # Find which nodes this operation uses
            operand_indices = torch.where(torch.abs(O_step) > 1e-6)[0]
            operand_coeffs = O_step[operand_indices]

            print(f"  Step {step}:")
            print(f"    G = {G_step} ({'log' if G_step == 0 else 'linear'})")
            print(f"    O_step_sum = {O_step_sum}")
            print(f"    Operand indices: {operand_indices.tolist()}")
            print(f"    Operand coefficients: {operand_coeffs.tolist()}")

            # Check masking decision
            is_identity = abs(O_step_sum - 1.0) < 1e-6
            print(f"    Masked as identity: {is_identity}")

            if not is_identity:
                print(f"    ✅ Real operation: {len(operand_indices)} operands")
            else:
                if len(operand_indices) == 1 and abs(operand_coeffs[0].item()) == 1.0:
                    print(
                        f"    ⚪ Identity operation: copying node {operand_indices[0].item()}"
                    )
                else:
                    print(
                        f"    ❓ Suspicious: marked as identity but has unusual structure"
                    )
            print()

        print()


def debug_without_masking():
    """Check what gate accuracy would be WITHOUT masking."""
    print("=== TESTING WITHOUT MASKING ===\n")

    # Get a batch of data
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=8,
        val_batch_size=8,
        max_depth=4,
        seed=123,
        max_digits=4,
        max_decimal_places=4,
        block_size=32,
    )

    texts, target_tensors, valid_mask = next(train_loader)

    # Extract all valid targets
    valid_positions = valid_mask.nonzero(as_tuple=False)

    all_target_G = []
    for batch_idx, token_idx in valid_positions:
        target_dict = target_tensors[batch_idx.item()][token_idx.item()]
        target_G = target_dict["target_G"]  # (dag_depth,)
        all_target_G.append(target_G)

    if all_target_G:
        stacked_target_G = torch.stack(all_target_G)  # (num_valid, dag_depth)

        # Flatten to get all gate targets (WITHOUT masking)
        all_gates_flat = stacked_target_G.flatten()

        print(f"Total gate targets (no masking): {len(all_gates_flat)}")
        print(f"Distribution: {torch.bincount(all_gates_flat.long()).tolist()}")

        num_ones = (all_gates_flat == 1).sum().item()
        num_zeros = (all_gates_flat == 0).sum().item()

        print(f"Proportion of 1s: {num_ones / len(all_gates_flat):.3f}")
        print(f"Proportion of 0s: {num_zeros / len(all_gates_flat):.3f}")

        # Test accuracy without masking
        # If model always predicts 1
        accuracy_always_1 = num_ones / len(all_gates_flat)
        print(f"Accuracy if always predicting 1: {accuracy_always_1:.3f}")

        # If model always predicts 0
        accuracy_always_0 = num_zeros / len(all_gates_flat)
        print(f"Accuracy if always predicting 0: {accuracy_always_0:.3f}")

        print("\n💡 If gate accuracy is stuck around the 'always predict 1' value,")
        print("   then the model learned the bias but masking prevents improvement!")


if __name__ == "__main__":
    debug_masking_logic()
    print("\n" + "=" * 60 + "\n")
    debug_without_masking()
