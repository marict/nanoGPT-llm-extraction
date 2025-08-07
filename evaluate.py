"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict

import tiktoken
import torch
import torch.distributed as dist

from data.dagset.heldout_expressions import heldout_segments
from data.dagset.streaming import (
    digit_onehot_to_float,
    expressions_to_tensors,
    tensor_to_expression,
)
from predictor_utils import compute_dag_loss, tokenize_texts


def _sharpen_digit_predictions(pred_digit_logits: torch.Tensor) -> torch.Tensor:
    """Convert raw digit logits to a discrete one-hot representation.

    Args:
        pred_digit_logits: Tensor of shape (num_nodes, D, base) with raw logits.
    Returns:
        Tensor of same shape containing 0/1 one-hot vectors.
    """
    # Apply softmax to get probabilities then take argmax
    digit_indices = torch.argmax(pred_digit_logits, dim=-1)  # (num_nodes, D)
    sharp = torch.zeros_like(pred_digit_logits)
    # Scatter 1.0 at argmax positions
    sharp.scatter_(-1, digit_indices.unsqueeze(-1), 1.0)
    return sharp


def _extract_initial_value(digit_data, sign, cfg, is_target: bool = True) -> float:
    """Extract initial value from digit data (target or predicted)."""
    if is_target:
        # Target digits are already one-hot
        digit_onehot = digit_data
    else:
        # Predicted digits need softmax -> argmax -> one-hot conversion
        digit_probs = torch.softmax(digit_data, dim=-1)
        digit_indices = torch.argmax(digit_probs, dim=1)
        digit_onehot = torch.zeros_like(digit_probs)
        for d, idx in enumerate(digit_indices):
            digit_onehot[d, idx] = 1.0
        sign = 1.0 if sign >= 0 else -1.0

    # Check if this initial node is used (has valid one-hot encoding)
    row_sums = digit_onehot.sum(dim=-1)
    if not (row_sums > 0).any():
        # This initial node is unused, return 0.0
        return 0.0

    return digit_onehot_to_float(
        digit_onehot, sign, cfg.max_digits, cfg.max_decimal_places
    )


def _print_initial_values_comparison(
    target_digits, target_V_sign, pred_digit_logits, pred_V_sign, cfg
):
    """Print comparison of target vs predicted initial values."""
    num_initial_nodes = target_digits.shape[0]
    print(f"\n--- Initial Values Comparison ---")

    for n in range(num_initial_nodes):
        target_val = _extract_initial_value(
            target_digits[n], target_V_sign[n].item(), cfg, is_target=True
        )
        pred_val = _extract_initial_value(
            pred_digit_logits[n], pred_V_sign[n].item(), cfg, is_target=False
        )

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


def _execute_dag_prediction(dag_executor, digit_logits, V_sign, O, G) -> float:

    # Add batch and time dimensions for executor (single sample)
    pred_digit_logits_exec = digit_logits.unsqueeze(0).unsqueeze(0)
    pred_V_sign_exec = V_sign.unsqueeze(0).unsqueeze(0)
    pred_O_exec = O.unsqueeze(0).unsqueeze(0)
    pred_G_exec = G.unsqueeze(0).unsqueeze(0)

    result = dag_executor(
        pred_digit_logits_exec, pred_V_sign_exec, pred_O_exec, pred_G_exec
    )
    return result[0, 0].item()


def _print_token_by_token_breakdown(
    texts, target_tensors, valid_mask, cfg, batch_idx=0
):
    """Print token-by-token breakdown showing what substring the model sees and target expression."""
    print(f"\n--- Token-by-Token Learning Objective ---")

    text = texts[batch_idx]
    sample_mask = valid_mask[batch_idx]  # (T,)
    sample_targets = target_tensors[batch_idx]  # List of target dicts

    print(f"\nFinal model input: '{text}'")

    # Track which token positions are valid for DAG prediction
    valid_positions = torch.where(sample_mask)[0]
    print(f"Valid positions: {valid_positions.tolist()}")

    print(f"\nTarget breakdown:")

    # Get tokenization
    enc = tiktoken.get_encoding("gpt2")
    actual_tokens = enc.encode_ordinary(text)
    actual_token_count = len(actual_tokens)

    # Show enough positions to include all valid positions, with a reasonable limit
    max_valid_pos = valid_positions[-1].item() if len(valid_positions) > 0 else 0
    display_limit = min(max(25, max_valid_pos + 1), len(sample_targets))

    def get_partial_text(pos):
        """Get the partial text up to position pos."""
        if pos + 1 <= actual_token_count:
            partial_tokens = actual_tokens[: pos + 1]
            return enc.decode(partial_tokens)
        else:
            return text[: pos + 1] if pos + 1 <= len(text) else text

    for pos in range(display_limit):
        if pos < actual_token_count:
            # This is within the actual tokenized text
            partial_text = get_partial_text(pos)
            is_valid = sample_mask[pos].item()

            if is_valid:
                # Get target expression for valid positions
                target_dict = sample_targets[pos]
                target_expr_sympy = tensor_to_expression(
                    target_dict["target_digits"],
                    target_dict["target_V_sign"],
                    target_dict["target_O"],
                    target_dict["target_G"],
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                )
                target_expr = str(target_expr_sympy)
                # Truncate very long expressions for readability
                if len(target_expr) > 60:
                    target_expr = target_expr[:57] + "..."

                print(
                    f"  Position {pos:2d}: '{partial_text}' → [VALID] Target = {target_expr}"
                )
            else:
                print(
                    f"  Position {pos:2d}: '{partial_text}' → [NOT VALID] (invalid substring)"
                )

    # Count padding beyond display limit
    padding_count = max(0, len(sample_targets) - display_limit)
    if padding_count > 0:
        print(f"  ... and {padding_count} tokens of padding")


def print_and_return_heldout_metrics(model, device):
    """Evaluate model on heldout expressions and compute accuracy metrics."""

    model.eval()

    segment_metrics = {}
    total_metrics = {
        "digit_accuracy": 0.0,
        "sign_accuracy": 0.0,
        "op_accuracy": 0.0,
        "gate_accuracy": 0.0,
        "count": 0,
    }

    print("\n=== HELDOUT EXPRESSIONS EVALUATION ===")
    with torch.no_grad():
        for segment, expressions in heldout_segments.items():
            segment_metrics[segment] = {
                "digit_accuracy": 0.0,
                "sign_accuracy": 0.0,
                "op_accuracy": 0.0,
                "gate_accuracy": 0.0,
                "count": 0,
            }

            print(f"\n--- {segment.upper()} ---")

            for str_expr, sympy_expr in expressions:
                tokens = tokenize_texts([str_expr], model.config.block_size, device)

                # Skip expressions that are too long.
                if len(tokens) >= model.config.block_size:
                    print(
                        f"Expression {str_expr} has more tokens than the model block size. Tokens: {len(tokens)}, Block size: {model.config.block_size}"
                    )
                    continue

                try:
                    # Get target tensors for this one expression
                    target_tensors = expressions_to_tensors(
                        [sympy_expr],
                        model.config.block_size,
                        depth=model.config.dag_depth,
                        max_digits=model.config.max_digits,
                        max_decimal_places=model.config.max_decimal_places,
                    )
                except Exception as e:
                    print(f"Error when converting expression: {str_expr}")
                    raise

                # Get predictions
                hidden_states = model.forward_hidden(tokens)
                pred_digit_logits, pred_V_sign, pred_O, pred_G = model.dag_predictor(
                    hidden_states
                )

                # Last non-zero token position is the only one we want scored.
                last_token_position = torch.where(tokens[-1] != 0.0)[0][-1]
                valid_mask = torch.zeros((1, tokens.shape[-1]), dtype=torch.bool)
                valid_mask[0, last_token_position] = True

                # Compute accuracies on heldout expression
                losses = compute_dag_loss(
                    pred_digit_logits,
                    pred_V_sign,
                    pred_O,
                    pred_G,
                    target_tensors,
                    model.dag_executor,
                    model.config,
                )

                # Extract accuracies
                digit_acc = losses["digit_accuracy"].item()
                sign_acc = losses["sign_accuracy"].item()
                op_acc = losses["op_accuracy"].item()
                gate_acc = losses["gate_accuracy"].item()

                # Print per-expression results
                print(
                    f"  {str_expr:<25} | Digits: {digit_acc:.1%} | Signs: {sign_acc:.1%} | Ops: {op_acc:.1%} | Gates: {gate_acc:.1%}"
                )

                # Accumulate segment metrics
                segment_metrics[segment]["digit_accuracy"] += digit_acc
                segment_metrics[segment]["sign_accuracy"] += sign_acc
                segment_metrics[segment]["op_accuracy"] += op_acc
                segment_metrics[segment]["gate_accuracy"] += gate_acc
                segment_metrics[segment]["count"] += 1

                # Accumulate total metrics
                total_metrics["digit_accuracy"] += digit_acc
                total_metrics["sign_accuracy"] += sign_acc
                total_metrics["op_accuracy"] += op_acc
                total_metrics["gate_accuracy"] += gate_acc
                total_metrics["total_expressions"] += 1

    # Print segment averages
    print(f"\n--- SEGMENT AVERAGES ---")
    for segment, metrics in segment_metrics.items():
        if metrics["count"] > 0:
            avg_digit = metrics["digit_accuracy"] / metrics["count"]
            avg_sign = metrics["sign_accuracy"] / metrics["count"]
            avg_op = metrics["op_accuracy"] / metrics["count"]
            avg_gate = metrics["gate_accuracy"] / metrics["count"]

            print(
                f"{segment:<15} | Digits: {avg_digit:.1%} | Signs: {avg_sign:.1%} | Ops: {avg_op:.1%} | Gates: {avg_gate:.1%}"
            )

    # Print overall average
    if total_metrics["total_expressions"] > 0:
        overall_digit = (
            total_metrics["digit_accuracy"] / total_metrics["total_expressions"]
        )
        overall_sign = (
            total_metrics["sign_accuracy"] / total_metrics["total_expressions"]
        )
        overall_op = total_metrics["op_accuracy"] / total_metrics["total_expressions"]
        overall_gate = (
            total_metrics["gate_accuracy"] / total_metrics["total_expressions"]
        )

        print(f"\n--- OVERALL AVERAGE ---")
        print(
            f"All Expressions | Digits: {overall_digit:.1%} | Signs: {overall_sign:.1%} | Ops: {overall_op:.1%} | Gates: {overall_gate:.1%}"
        )
        print(f"Total expressions evaluated: {total_metrics['total_expressions']}")

        return {
            "heldout/digit_accuracy": overall_digit,
            "heldout/sign_accuracy": overall_sign,
            "heldout/op_accuracy": overall_op,
            "heldout/gate_accuracy": overall_gate,
        }
    else:
        raise ValueError("No expressions evaluated")


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

    # Print token-by-token breakdown of learning objective
    _print_token_by_token_breakdown(texts, target_tensors, valid_mask, cfg, batch_idx=0)

    # Get the last valid token position for this sample (complete expression)
    sample_mask = valid_mask[0]  # (T,)
    valid_positions = torch.where(sample_mask)[0]
    if len(valid_positions) == 0:
        raise ValueError(f"No valid tokens found in validation sample: '{texts[0]}'")
    last_valid_pos = valid_positions[-1].item()

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

    # Sharpen predictions for cleaner expression display and consistent execution
    # Note: V_sign, O, and G are already sharpened by STE in the predictor forward pass
    sharp_digit_logits = _sharpen_digit_predictions(single_digit_logits)

    # Convert predicted tensors to expression string with sharpened values
    pred_expr = tensor_to_expression(
        sharp_digit_logits,
        single_V_sign,
        single_O,
        single_G,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Print initial values comparison
    _print_initial_values_comparison(
        target_digits, target_V_sign, single_digit_logits, single_V_sign, cfg
    )

    # Execute predicted DAG
    pred_final_exec = _execute_dag_prediction(
        dag_executor, sharp_digit_logits, single_V_sign, single_O, single_G
    )

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
    """Run evaluation on *eval_iters* batches and return aggregated metrics."""
    model.eval()
    _eval_random.seed(seed)

    total_losses = {
        k: 0.0
        for k in (
            "total_loss",
            "digit_loss",
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
        for i, (texts, target_tensors) in enumerate(val_loader):
            if i >= eval_iters:
                break

            for key, tensor in target_tensors.items():
                if isinstance(tensor, torch.Tensor):
                    target_tensors[key] = tensor.to(device)

            valid_mask = target_tensors["valid_mask"]

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

            # Calculate expression-level valid rate (excluding padding)
            expression_valid_tokens = valid_mask.sum().item()
            expression_total_tokens = target_tensors["total_expressions"].sum().item()

            if expression_total_tokens > 0:
                total_metrics["expression_valid_rate"] += (
                    expression_valid_tokens / expression_total_tokens
                )

            num_batches += 1

            if master_process:
                sample_text = texts[0] if texts else "N/A"
                sample_valid_rate = (
                    valid_mask[0].float().mean().item() if len(valid_mask) > 0 else 0.0
                )
                print(
                    f'[val] batch {i}: "{sample_text[:50]}...", valid_rate={sample_valid_rate:.1%}'
                )

                # Display detailed validation for every batch during validation sessions
                if len(texts) > 0:
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

            if num_batches >= eval_iters:
                break

        import pdb

        pdb.set_trace()

        # Run heldout expressions evaluations
        # NOTE: This is disabled for now because it's not working.
        heldout_metrics = print_and_return_heldout_metrics(model, device)

        import pdb

        pdb.set_trace()

    # Average metrics over successful batches
    if num_batches > 0:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
