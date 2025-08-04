"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict

import sympy
import torch
import torch.distributed as dist

from data.dagset.streaming import digit_onehot_to_float, tensor_to_expression
from predictor_utils import compute_dag_loss, tokenize_texts


def print_detailed_validation_sample(
    texts,
    target_tensors,
    valid_mask,
    pred_digit_logits,
    pred_V_sign,
    pred_O,
    pred_G,
    dag_executor,
    cfg,
    seed,
    batch_idx,
):
    """
    Print detailed validation information for a single sample.

    Args:
        texts: List of input text expressions
        target_tensors: Target tensor data for the batch
        valid_mask: Boolean mask indicating valid positions
        pred_digit_logits: Predicted digit logits
        pred_V_sign: Predicted signs
        pred_O: Predicted operand selectors
        pred_G: Predicted domain gates
        dag_predictor: DAG predictor model
        dag_executor: DAG executor for computing final values
        cfg: Configuration object with max_digits, max_decimal_places, batch_size
        seed: Random seed used for validation
        batch_idx: Index of current batch
    """
    # Reconstruct seed for this sample
    sample_seed = seed + batch_idx * cfg.batch_size + 10000  # Validation offset

    print(f"\n=== DETAILED VALIDATION SAMPLE (seed={sample_seed}) ===")
    print(f"Expression Text: '{texts[0]}'")

    # Get the last valid token position for this sample (complete expression)
    sample_mask = valid_mask[0]  # (T,)
    last_valid_pos = -1
    for pos in range(len(sample_mask)):
        if sample_mask[pos]:
            last_valid_pos = pos

    if last_valid_pos < 0:
        raise ValueError(
            f"No valid tokens found in validation sample: '{texts[0]}'. This should never happen during validation."
        )

    # Get target and prediction for the final/complete expression
    target_dict = target_tensors[0][last_valid_pos]

    # Target DAG tensors - convert from digit targets
    target_digits = target_dict["target_digits"]  # (num_initial_nodes, D, base)
    target_V_sign = target_dict["target_V_sign"]  # (total_nodes,)
    target_O = target_dict["target_O"]  # (dag_depth, total_nodes)
    target_G = target_dict["target_G"]  # (dag_depth,)
    target_final_exec = target_dict["target_final_exec"]

    # Convert target tensors to expression string
    target_expr = tensor_to_expression(
        target_digits,
        target_V_sign,
        target_O,
        target_G,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Extract single prediction tensors for expression conversion
    single_digit_logits = pred_digit_logits[
        0, last_valid_pos
    ]  # (num_initial_nodes, D, base)
    single_V_sign = pred_V_sign[0, last_valid_pos]  # (total_nodes,)
    single_O = pred_O[0, last_valid_pos]  # (dag_depth, total_nodes)
    single_G = pred_G[0, last_valid_pos]  # (dag_depth,)

    # Sharpen predictions to discrete choices for cleaner expression display
    sharp_digit_logits = single_digit_logits.clone()
    sharp_V_sign = single_V_sign.clone()
    sharp_O = torch.zeros_like(single_O)
    sharp_G = (single_G > 0.5).float()  # Binary threshold for domain gates

    # For O tensor: Use threshold-based approach to preserve unary/binary/trinary operations
    dag_depth, _ = single_O.shape
    threshold = 0.1  # Only keep operands with significant probability (>10% confidence)

    for step in range(dag_depth):
        step_coeffs = single_O[step]  # (total_nodes,)

        # Find operands above threshold
        significant_mask = torch.abs(step_coeffs) > threshold
        significant_indices = torch.nonzero(significant_mask, as_tuple=False).flatten()

        if len(significant_indices) > 0:
            # Keep the signs and normalize to clean values (+1, -1)
            for idx in significant_indices:
                coeff_val = step_coeffs[idx].item()
                # Preserve sign but make it clean: positive -> +1, negative -> -1
                sharp_O[step, idx] = 1.0 if coeff_val > 0 else -1.0

    # Convert predicted tensors to expression string with sharpened values
    pred_expr = tensor_to_expression(
        sharp_digit_logits,
        sharp_V_sign,
        sharp_O,
        sharp_G,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Extract and log initial values for comparison
    num_initial_nodes = target_digits.shape[0]

    print(f"\n--- Initial Values Comparison ---")

    # Target initial values (from target_digits)
    target_initial_values = []
    for n in range(num_initial_nodes):
        target_digit_onehot = target_digits[n]  # (D, base)
        target_sign = target_V_sign[n].item()
        try:
            target_value = digit_onehot_to_float(
                target_digit_onehot, target_sign, cfg.max_digits, cfg.max_decimal_places
            )
            target_initial_values.append(target_value)
        except Exception as e:
            print(f"Error converting target digit {n}: {e}")
            target_initial_values.append(float("nan"))

    # Predicted initial values (from predicted digit logits)
    pred_initial_values = []
    for n in range(num_initial_nodes):
        pred_digit_logits_node = single_digit_logits[n]  # (D, base)
        pred_sign = single_V_sign[n].item()

        try:
            # Convert logits to probabilities and then to one-hot (same as tensor_to_expression)
            digit_probs = torch.softmax(pred_digit_logits_node, dim=-1)
            digit_indices = torch.argmax(digit_probs, dim=1)

            # Create one-hot encoding
            digit_onehot = torch.zeros_like(digit_probs)
            for d, idx in enumerate(digit_indices):
                digit_onehot[d, idx] = 1.0

            # Convert sign to discrete
            sign = 1.0 if pred_sign >= 0 else -1.0

            # Convert to float value
            pred_value = digit_onehot_to_float(
                digit_onehot, sign, cfg.max_digits, cfg.max_decimal_places
            )
            pred_initial_values.append(pred_value)
        except Exception as e:
            print(f"Error converting predicted digit {n}: {e}")
            pred_initial_values.append(float("nan"))

    # Print initial values comparison
    for n in range(num_initial_nodes):
        target_val = target_initial_values[n]
        pred_val = pred_initial_values[n]
        error = (
            abs(target_val - pred_val)
            if not (
                torch.isnan(torch.tensor(target_val))
                or torch.isnan(torch.tensor(pred_val))
            )
            else float("nan")
        )
        print(
            f"Initial[{n}]: Target={target_val:.6f}, Predicted={pred_val:.6f}, Error={error:.6f}"
        )

    # Execute predicted DAG if executor is available
    pred_final_exec = 0.0
    if dag_executor is not None:
        # Reuse the already extracted single tensors for execution
        # Add batch and time dimensions for executor
        pred_digit_logits_exec = single_digit_logits.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, num_initial_nodes, D, base)
        pred_V_sign_exec = single_V_sign.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, total_nodes)
        pred_O_exec = single_O.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, dag_depth, total_nodes)
        pred_G_exec = single_G.unsqueeze(0).unsqueeze(0)  # (1, 1, dag_depth)

        pred_final_exec = dag_executor(
            pred_digit_logits_exec,
            pred_V_sign_exec,
            pred_O_exec,
            pred_G_exec,
        )[0, 0].item()

    print(f"\n--- Expression Comparison ---")
    print(f"Target Expression: {str(target_expr)}")
    print(f"Target Final Execution: {target_final_exec:.6f}")
    print(f"Predicted Expression: {str(pred_expr)}")
    print(f"Predicted Final Execution: {pred_final_exec:.6f}")

    if not torch.isnan(torch.tensor(pred_final_exec)):
        exec_error = abs(target_final_exec - pred_final_exec)
        print(f"Execution Error: {exec_error:.6f}")
    else:
        print("Execution Error: NaN (execution failed)")

    print("=" * 60 + "\n")


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
            "digit_loss",
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
        "digit_accuracy": 0.0,
        "sign_accuracy": 0.0,
        "op_accuracy": 0.0,
        "gate_accuracy": 0.0,
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

                    pred_digit_logits, pred_V_sign, pred_O, pred_G = dag_predictor(
                        hidden_states
                    )

                # Compute loss using DAG tensor loss function
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

                # Accumulate losses
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()

                # Accumulate metrics (including accuracy metrics)
                for key in losses.keys():
                    if key in total_metrics:
                        total_metrics[key] += losses[key].item()

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

                if master_process:
                    sample_text = texts[0] if texts else "N/A"
                    sample_valid_rate = (
                        valid_mask[0].float().mean().item()
                        if len(valid_mask) > 0
                        else 0.0
                    )
                    print(
                        f'[val] batch {i}: "{sample_text[:50]}...", valid_rate={sample_valid_rate:.1%}'
                    )

                    # Display detailed validation for every batch during validation sessions
                    if valid_tokens_count > 0 and len(texts) > 0:
                        print_detailed_validation_sample(
                            texts=texts,
                            target_tensors=target_tensors,
                            valid_mask=valid_mask,
                            pred_digit_logits=pred_digit_logits,
                            pred_V_sign=pred_V_sign,
                            pred_O=pred_O,
                            pred_G=pred_G,
                            dag_executor=dag_executor,
                            cfg=cfg,
                            seed=seed,
                            batch_idx=i,
                        )

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
