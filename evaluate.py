"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from predictor_utils import compute_dag_loss, tokenize_texts


def dag_tensors_to_english(V_mag, V_sign, O, G, dag_depth=None):
    """
    Convert DAG tensors to readable English representation using argmax.

    Args:
        V_mag: (total_nodes,) magnitude values
        V_sign: (total_nodes,) sign values
        O: (dag_depth, total_nodes) operand selectors
        G: (dag_depth,) domain gates
        dag_depth: Number of operation steps (default: use O.shape[0])

    Returns:
        str: Human-readable representation of the DAG
    """
    if dag_depth is None:
        dag_depth = O.shape[0]

    total_nodes = V_mag.shape[0]
    initial_slots = total_nodes // 2  # Half for initial values, half for intermediate

    # Format initial values
    initial_values = []
    for i in range(initial_slots):
        mag = V_mag[i].item()
        sign = V_sign[i].item()
        value = sign * mag
        initial_values.append(f"{value:.3f}")

    result = f"Initial Values: [{', '.join(initial_values)}]\n"

    # Format operations
    operations = []
    for step in range(dag_depth):
        # Get operand selection (argmax over operands)
        operand_probs = O[step]  # (total_nodes,)

        # Find which operands are selected (non-zero after argmax)
        # For simplicity, take top 2 operands as this is typical for binary operations
        _, top_indices = torch.topk(operand_probs, k=min(2, total_nodes))

        # Domain gate (sigmoid for interpretation)
        domain = torch.sigmoid(G[step]).item()
        domain_str = "linear" if domain > 0.5 else "log"

        # Result slot
        result_slot = initial_slots + step
        result_mag = V_mag[result_slot].item() if result_slot < total_nodes else 0.0
        result_sign = V_sign[result_slot].item() if result_slot < total_nodes else 1.0
        result_value = result_sign * result_mag

        # Format operands
        operand_strs = []
        for idx in top_indices:
            idx_val = idx.item()
            if idx_val < total_nodes:
                op_mag = V_mag[idx_val].item()
                op_sign = V_sign[idx_val].item()
                op_value = op_sign * op_mag
                operand_strs.append(f"slot{idx_val}({op_value:.3f})")

        op_str = f"Step {step}: {' ○ '.join(operand_strs)} → slot{result_slot}({result_value:.3f}) [{domain_str}]"
        operations.append(op_str)

    result += "Operations:\n" + "\n".join(f"  {op}" for op in operations)

    # Final result (last slot)
    final_slot = total_nodes - 1
    final_mag = V_mag[final_slot].item()
    final_sign = V_sign[final_slot].item()
    final_value = final_sign * final_mag
    result += f"\nFinal Result: {final_value:.6f}"

    return result


def evaluate_dag_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    ctx,
    cfg,
    eval_iters: int,
    seed: int,
) -> Dict[str, float]:
    """Run evaluation on *eval_iters* batches and return aggregated metrics.

    This function works with the new per-token streaming.py data format.
    """
    model.eval()
    _eval_random.seed(seed)

    total_losses = {
        k: 0.0
        for k in (
            "total_loss",
            "V_mag_loss",
            "V_sign_loss",
            "O_loss",
            "G_loss",
            "exec_loss",
        )
    }
    total_metrics = {
        "valid_tokens": 0.0,
        "total_tokens": 0.0,
        "expression_valid_rate": 0.0,
    }

    num_batches = 0
    # For evaluation printing, we want to show details in most cases
    # Only skip if we're explicitly in a distributed non-master process
    try:
        master_process = not dist.is_initialized() or dist.get_rank() == 0
    except:
        master_process = True  # Default to showing details

    with torch.no_grad():
        for i, (texts, target_tensors, valid_mask) in enumerate(val_loader):
            if i >= eval_iters:
                break

            try:

                for seq_targets in target_tensors:
                    for target_dict in seq_targets:
                        for key, tensor in target_dict.items():
                            if isinstance(tensor, torch.Tensor):
                                target_dict[key] = tensor.to(device)

                valid_mask = valid_mask.to(device)

                # Inputs
                input_tokens = tokenize_texts(texts, cfg.block_size, device)

                # Model forward pass with DAG model
                with ctx:

                    hidden_states = model.forward_hidden(input_tokens)

                    dag_predictor = model.dag_predictor
                    if dag_predictor is None:
                        raise RuntimeError("Model has no DAG predictor (dag_depth=0)")

                    pred_V_mag, pred_V_sign, pred_O, pred_G = dag_predictor(
                        hidden_states
                    )

                # Compute loss using DAG tensor loss function
                dag_executor = getattr(model, "dag_executor", None)
                losses = compute_dag_loss(
                    pred_V_mag,
                    pred_V_sign,
                    pred_O,
                    pred_G,
                    target_tensors,
                    valid_mask,
                    dag_executor=dag_executor,
                )

                # Accumulate losses
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()

                # Compute per-token metrics
                valid_tokens_count = valid_mask.sum().item()

                # Calculate expression-level valid rate (excluding padding)
                batch_size, seq_len = valid_mask.shape
                expression_valid_tokens = 0
                expression_total_tokens = 0

                for b in range(batch_size):
                    # Find the last non-padding token (expression length)
                    sequence_mask = valid_mask[b]  # (seq_len,)

                    # Find expression length by looking for the transition to padding
                    # Padding is always False values at the end
                    expression_length = seq_len
                    for i in range(seq_len - 1, -1, -1):
                        if (
                            i == 0
                            or sequence_mask[i]
                            or (i < seq_len - 1 and sequence_mask[i + 1])
                        ):
                            expression_length = i + 1
                            break

                    # Count valid tokens within the expression (before padding)
                    expression_tokens = sequence_mask[:expression_length]
                    expression_valid_tokens += expression_tokens.sum().item()
                    expression_total_tokens += expression_length

                if expression_total_tokens > 0:
                    total_metrics["expression_valid_rate"] += (
                        expression_valid_tokens / expression_total_tokens
                    )

                num_batches += 1

                if master_process and (i % 10 == 0 or i < 3):
                    sample_text = texts[0] if texts else "N/A"
                    sample_valid_rate = (
                        valid_mask[0].float().mean().item()
                        if len(valid_mask) > 0
                        else 0.0
                    )
                    print(
                        f'[val] batch {i}: "{sample_text[:50]}...", valid_rate={sample_valid_rate:.1%}'
                    )

                    # Display example every 10 batches or first 3 batches
                    if (
                        (i % 10 == 0 or i < 3)
                        and valid_tokens_count > 0
                        and len(texts) > 0
                    ):
                        # Reconstruct seed for this sample
                        sample_seed = (
                            seed + i * cfg.batch_size + 10000
                        )  # Validation offset

                        print(
                            f"\n=== DETAILED VALIDATION SAMPLE (seed={sample_seed}) ==="
                        )
                        print(f"Expression Text: '{texts[0]}'")

                        # Get the last valid token position for this sample (complete expression)
                        sample_mask = valid_mask[0]  # (T,)
                        last_valid_pos = -1
                        for pos in range(len(sample_mask)):
                            if sample_mask[pos]:
                                last_valid_pos = pos

                        if last_valid_pos >= 0:
                            # Get target and prediction for the final/complete expression
                            target_dict = target_tensors[0][last_valid_pos]

                            # Target DAG tensors
                            target_V_mag = target_dict["target_V_mag"]  # (total_nodes,)
                            target_V_sign = target_dict[
                                "target_V_sign"
                            ]  # (total_nodes,)
                            target_O = target_dict[
                                "target_O"
                            ]  # (dag_depth, total_nodes)
                            target_G = target_dict["target_G"]  # (dag_depth,)
                            target_final_exec = target_dict["target_final_exec"]

                            # Predicted DAG tensors (from the forward pass)
                            pred_V_mag_sample = pred_V_mag[
                                0, last_valid_pos
                            ]  # (total_nodes,)
                            pred_V_sign_sample = pred_V_sign[
                                0, last_valid_pos
                            ]  # (total_nodes,)
                            pred_O_sample = pred_O[
                                0, last_valid_pos
                            ]  # (dag_depth, total_nodes)
                            pred_G_sample = pred_G[0, last_valid_pos]  # (dag_depth,)

                            # Convert to English representations
                            target_english = dag_tensors_to_english(
                                target_V_mag, target_V_sign, target_O, target_G
                            )
                            pred_english = dag_tensors_to_english(
                                pred_V_mag_sample,
                                pred_V_sign_sample,
                                pred_O_sample,
                                pred_G_sample,
                            )

                            # Execute predicted DAG if executor is available
                            pred_final_exec = 0.0
                            if dag_executor is not None:
                                try:
                                    # Add batch and time dimensions for executor
                                    pred_V_mag_exec = pred_V_mag_sample.unsqueeze(
                                        0
                                    ).unsqueeze(
                                        0
                                    )  # (1, 1, total_nodes)
                                    pred_V_sign_exec = pred_V_sign_sample.unsqueeze(
                                        0
                                    ).unsqueeze(
                                        0
                                    )  # (1, 1, total_nodes)
                                    pred_O_exec = pred_O_sample.unsqueeze(0).unsqueeze(
                                        0
                                    )  # (1, 1, dag_depth, total_nodes)
                                    pred_G_exec = pred_G_sample.unsqueeze(0).unsqueeze(
                                        0
                                    )  # (1, 1, dag_depth)

                                    pred_final_exec = dag_executor(
                                        pred_V_mag_exec,
                                        pred_V_sign_exec,
                                        pred_O_exec,
                                        pred_G_exec,
                                    )[0, 0].item()
                                except Exception as e:
                                    pred_final_exec = float("nan")
                                    print(f"  Warning: DAG execution failed: {e}")

                            print("\n--- TARGET DAG ---")
                            print(target_english)
                            print(f"Target Final Execution: {target_final_exec:.6f}")

                            print("\n--- PREDICTED DAG ---")
                            print(pred_english)
                            print(f"Predicted Final Execution: {pred_final_exec:.6f}")

                            if not torch.isnan(torch.tensor(pred_final_exec)):
                                exec_error = abs(target_final_exec - pred_final_exec)
                                print(f"Execution Error: {exec_error:.6f}")
                            else:
                                print("Execution Error: NaN (execution failed)")

                        print("=" * 60 + "\n")

            except Exception as e:
                # Fail fast on ALL exceptions during evaluation - no skipping
                if master_process:
                    print(f"Error in evaluation batch {i+1}/{eval_iters}: {e}")
                raise

            if num_batches >= eval_iters:
                break

    # Average metrics over successful batches
    if num_batches > 0:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
