"""Propose a mask-based solution for the gate accuracy issue."""

import torch


def create_gate_mask(target_tensors, dag_depth):
    """Create a mask indicating which gate positions are actually used.

    This analyzes the target tensors to determine which gate positions
    correspond to actual operations vs. unused padding positions.
    """
    batch_size = len(target_tensors)
    gate_mask = torch.zeros(batch_size, dag_depth, dtype=torch.bool)

    for b, target_list in enumerate(target_tensors):
        for t, target_dict in enumerate(target_list):
            if target_dict is not None:
                # Analyze the O tensor to see which operations are actually used
                target_O = target_dict["target_O"]  # (dag_depth, total_nodes)

                for step in range(dag_depth):
                    # Check if this step has any non-zero operand coefficients
                    # (indicating an actual operation happens at this step)
                    if torch.any(target_O[step] != 0):
                        gate_mask[b, step] = True

                break  # Only need to check one valid position per batch

    return gate_mask


def compute_masked_gate_loss_and_accuracy(pred_G_valid, target_G, gate_mask):
    """Compute gate loss and accuracy only for actually used gate positions."""

    # Apply sigmoid to predictions
    pred_G_sigmoid = torch.sigmoid(pred_G_valid)

    # Create mask for valid positions (flatten to match pred/target shape)
    flat_mask = gate_mask.view(-1)  # (batch_size * dag_depth,)

    if flat_mask.any():
        # Only compute loss on masked positions
        masked_pred = pred_G_sigmoid[flat_mask]
        masked_target = target_G[flat_mask]

        # Gate loss (only on meaningful positions)
        G_loss = torch.nn.functional.mse_loss(masked_pred, masked_target)

        # Gate accuracy (only on meaningful positions)
        pred_G_discrete = (masked_pred > 0.5).float()
        gate_correct = (pred_G_discrete == masked_target).float()
        gate_accuracy = gate_correct.mean()
    else:
        # No valid gates - return zero loss and perfect accuracy
        G_loss = torch.tensor(0.0, device=pred_G_valid.device)
        gate_accuracy = torch.tensor(1.0, device=pred_G_valid.device)

    return G_loss, gate_accuracy


def test_mask_based_solution():
    """Test the mask-based solution concept."""

    print("=== Testing Mask-Based Gate Loss Solution ===")

    # Simulate a batch where only some gate positions are used
    batch_size = 2
    dag_depth = 4

    # Create mock target tensors
    target_tensors = []
    for b in range(batch_size):
        target_list = []
        # Simulate an expression that uses only 2 operations (steps 0 and 1)
        target_O = torch.zeros(dag_depth, 5)  # (dag_depth, total_nodes)
        target_O[0, 0] = 1.0  # Step 0 uses node 0
        target_O[0, 1] = 1.0  # Step 0 uses node 1
        target_O[1, 2] = 1.0  # Step 1 uses node 2
        # Steps 2 and 3 are unused (all zeros)

        target_dict = {
            "target_O": target_O,
            "target_G": torch.tensor([0.0, 1.0, 1.0, 1.0]),  # Only first 2 meaningful
        }
        target_list.append(target_dict)
        target_tensors.append(target_list)

    # Create gate mask
    gate_mask = create_gate_mask(target_tensors, dag_depth)
    print(f"Gate mask shape: {gate_mask.shape}")
    print(f"Gate mask: {gate_mask}")

    # Simulate predictions
    pred_G_valid = torch.tensor(
        [
            [0.1, 2.0, 0.8, 0.9],  # Batch 0: pred [0,1,1,1], target [0,1,1,1]
            [-1.0, 3.0, 0.7, 0.8],  # Batch 1: pred [0,1,1,1], target [0,1,1,1]
        ]
    )  # (batch_size, dag_depth)

    target_G = torch.tensor(
        [[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]  # Batch 0  # Batch 1
    )  # (batch_size, dag_depth)

    # Flatten for loss computation
    pred_G_flat = pred_G_valid.view(-1)
    target_G_flat = target_G.view(-1)

    print(f"\n=== MASKED APPROACH (used positions only) ===")
    loss_new, accuracy_new = compute_masked_gate_loss_and_accuracy(
        pred_G_flat, target_G_flat, gate_mask
    )
    print(f"Mask:        {gate_mask.view(-1).tolist()}")
    print(f"Used positions only - Accuracy: {accuracy_new:.3f}")

    # Show which positions were actually evaluated
    flat_mask = gate_mask.view(-1)
    if flat_mask.any():
        masked_pred_sigmoid = torch.sigmoid(pred_G_flat[flat_mask])
        masked_pred_discrete = (masked_pred_sigmoid > 0.5).float()
        masked_target = target_G_flat[flat_mask]
        print(f"Masked preds: {masked_pred_discrete.tolist()}")
        print(f"Masked targets: {masked_target.tolist()}")

    print(f"\n=== RESULT ===")
    print(f"Masked accuracy (meaningful): {accuracy_new:.1%}")
    print("✅ Mask-based approach evaluates only meaningful positions!")


if __name__ == "__main__":
    test_mask_based_solution()

    print(f"\n=== NEXT STEPS ===")
    print("1. Integrate this masking into predictor_utils.py")
    print("2. Update _compute_g_loss to use the mask")
    print("3. Test that training shows meaningful gate accuracy progression")
    print("4. Verify that DAG execution still works correctly")
