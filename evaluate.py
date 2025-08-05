"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict

import tiktoken
import torch
import torch.distributed as dist

from data.dagset.streaming import digit_onehot_to_float, tensor_to_expression
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


def _sharpen_sign_predictions(pred_V_sign: torch.Tensor) -> torch.Tensor:
    """Sharpen sign predictions to exactly -1 or 1."""
    return torch.where(
        pred_V_sign >= 0,
        torch.tensor(1.0, device=pred_V_sign.device),
        torch.tensor(-1.0, device=pred_V_sign.device),
    )


def _sharpen_operand_predictions(
    pred_O: torch.Tensor, threshold: float = 0.1
) -> torch.Tensor:
    """Sharpen operand predictions to discrete {-1, 0, 1} values."""
    sharp_O = torch.zeros_like(pred_O)
    dag_depth, _ = pred_O.shape

    for step in range(dag_depth):
        step_coeffs = pred_O[step]
        significant_mask = torch.abs(step_coeffs) > threshold
        significant_indices = torch.where(significant_mask)[0]

        for idx in significant_indices:
            coeff_val = step_coeffs[idx].item()
            sharp_O[step, idx] = 1.0 if coeff_val > 0 else -1.0

    return sharp_O


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
    """Execute DAG with proper dimension handling."""
    if dag_executor is None:
        return 0.0

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
    sharp_digit_logits = _sharpen_digit_predictions(single_digit_logits)
    sharp_V_sign = _sharpen_sign_predictions(single_V_sign)
    sharp_O = _sharpen_operand_predictions(single_O)
    sharp_G = (single_G > 0.5).float()

    # Convert predicted tensors to expression string with sharpened values
    pred_expr = tensor_to_expression(
        sharp_digit_logits,
        sharp_V_sign,
        sharp_O,
        sharp_G,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # Print initial values comparison
    _print_initial_values_comparison(
        target_digits, target_V_sign, single_digit_logits, single_V_sign, cfg
    )

    # Execute predicted DAG
    pred_final_exec = _execute_dag_prediction(
        dag_executor, sharp_digit_logits, sharp_V_sign, sharp_O, sharp_G
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
