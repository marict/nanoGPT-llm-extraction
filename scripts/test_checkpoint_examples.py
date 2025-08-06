#!/usr/bin/env python3
"""
Script to test any checkpoint on DAG examples.

This script:
1. Tries to load an existing checkpoint (or creates a mock one)
2. Tests the checkpoint on simple DAG examples like 1.0 + 1.0
3. Shows the model's predictions vs expected results
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import sympy

from checkpoint_manager import CheckpointManager
from evaluate import (
    _extract_initial_value,
    _sharpen_digit_predictions,
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
    # Take the result from the last position (should be [0, -1] for single sample)
    final_result = result[0, -1].item()

    # Debug: If result is 0.0, show internal state
    if abs(final_result) < 1e-10:
        print(f"     🔍 DEBUG: Zero result detected, showing internal state:")
        print(f"       - Initial V_sign: {V_sign.tolist()}")
        print(f"       - Initial O (operands): {O.tolist()}")
        print(f"       - Initial G (gates): {G.tolist()}")
        print(f"       - Raw result tensor shape: {result.shape}")
        print(f"       - Raw result values: {result[0, -1].tolist()}")

        # Show the intermediate computation steps
        with torch.no_grad():
            # Convert digits to magnitudes
            initial_V_mag = dag_executor.digits_to_vmag(
                pred_digit_logits_exec, dag_executor.max_digits, dag_executor.base
            )
            print(f"       - Initial V_mag: {initial_V_mag[0, 0].tolist()}")

            # Show what the final computation should be
            final_idx = dag_executor.total_nodes - 1
            print(f"       - Final node index: {final_idx}")
            print(f"       - Total nodes: {dag_executor.total_nodes}")

    return final_result


def _print_predicted_initial_values(pred_digit_logits, pred_V_sign, cfg):
    """Print the predicted initial values for the DAG."""
    num_initial_nodes = pred_digit_logits.shape[0]
    print(f"     --- Predicted Initial Values ---")

    for n in range(num_initial_nodes):
        pred_val = _extract_initial_value(
            pred_digit_logits[n], pred_V_sign[n].item(), cfg, is_target=False
        )
        print(f"     Initial[{n}]: {pred_val:.6f}")


def _print_digit_probabilities(pred_digit_logits, node_idx=0):
    """Print digit probabilities for a given node to show model confidence."""
    print(f"     --- Digit Probabilities (Node {node_idx}) ---")

    # Get digit probabilities for the specified node
    node_digit_logits = pred_digit_logits[node_idx]  # (D, base)
    digit_probs = torch.softmax(node_digit_logits, dim=-1)  # (D, base)

    # Print probabilities for each digit position
    for digit_pos in range(digit_probs.shape[0]):
        probs = digit_probs[digit_pos]  # (base,)
        max_prob = probs.max().item()
        max_digit = probs.argmax().item()

        print(f"     Digit[{digit_pos}]: max={max_digit} (prob={max_prob:.3f})")

        # Show top 3 probabilities for this digit position
        top_probs, top_digits = torch.topk(probs, min(3, len(probs)))
        for i, (prob, digit) in enumerate(zip(top_probs, top_digits)):
            if i == 0:
                print(f"       Top: {digit.item()}={prob:.3f}")
            else:
                print(f"            {digit.item()}={prob:.3f}")


from data.dagset.streaming import expressions_to_tensors, tensor_to_expression
from models.dag_model import GPT, DAGExecutor, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel
from predictor_utils import tokenize_texts


def load_checkpoint(checkpoint_path: Path):
    """Load a checkpoint from a local path or W&B."""
    checkpoint_manager = CheckpointManager()

    # Check if it's a W&B path (starts with wandb://)
    if str(checkpoint_path).startswith("wandb://"):
        print(f"🔄 Downloading checkpoint from W&B: {checkpoint_path}")
        local_dir = Path("wandb_checkpoints")

        try:
            # Extract the W&B path from the wandb:// prefix
            wandb_path = str(checkpoint_path)[7:]
            downloaded_path = checkpoint_manager.download_checkpoint_from_wandb(
                local_dir=local_dir, run_name=wandb_path
            )
            if downloaded_path and downloaded_path.exists():
                print(f"✅ Successfully downloaded checkpoint to: {downloaded_path}")
                checkpoint_path = downloaded_path
            else:
                raise FileNotFoundError("Failed to download checkpoint from W&B")
        except Exception as e:
            print(f"❌ Error downloading from W&B: {e}")
            raise FileNotFoundError(f"Failed to download checkpoint from W&B: {e}")

    # Load the checkpoint
    if checkpoint_path.exists():
        print(f"🔄 Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = checkpoint_manager.load_checkpoint_from_path(checkpoint_path)

            # Handle different checkpoint formats
            if "model_args" in checkpoint:
                model_args = checkpoint["model_args"]
            elif "model_config" in checkpoint:
                # Convert dictionary to config object
                config_dict = checkpoint["model_config"]
                if "dag_depth" in config_dict and config_dict["dag_depth"] > 0:
                    # Filter out extra fields for PredictorOnlyConfig
                    valid_fields = [
                        "vocab_size",
                        "n_layer",
                        "n_embd",
                        "n_head",
                        "dropout",
                        "bias",
                        "dag_depth",
                        "block_size",
                        "max_digits",
                        "max_decimal_places",
                    ]
                    filtered_config = {
                        k: v for k, v in config_dict.items() if k in valid_fields
                    }
                    model_args = PredictorOnlyConfig(**filtered_config)
                else:
                    model_args = GPTConfig(**config_dict)
            else:
                raise KeyError(
                    "Neither 'model_args' nor 'model_config' found in checkpoint"
                )

            # Create model
            if isinstance(model_args, PredictorOnlyConfig):
                model = PredictorOnlyModel(model_args)
                model_type = "DAG Predictor"
            else:
                model = GPT(model_args)
                model_type = "DAG-GPT"

            # Load state dict
            model.load_state_dict(checkpoint["model"])
            print(f"✅ {model_type} model loaded successfully")
            print(
                f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}"
            )

            return model, model_args, checkpoint

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise e

    # If we get here, the checkpoint file doesn't exist
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def test_dag_examples(model, config):
    """Test the model on various DAG examples."""
    print(f"\n🧪 Testing DAG examples...")

    model.eval()

    # Test examples - use expressions that match the training data format
    test_expressions = [
        # Single expressions
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        # Simple expressions
        "1.0 + 1.0",
        "2.0 * 3.0",
        "10.0 - 5.0",
        "15.0 / 3.0",
        # More complex expressions like training data
        "-2.0 + 5.0",
        "69.919 - 7.211",
        "-907.0 + 7.211",
        "2.0 * 3.0 + 4.0",
        "10.0 - 5.0 / 2.0",
        "1.5 * 2.0 + 3.0",
        "8.0 / 2.0 - 1.0",
        # Very complex expressions
        "-1487.475/(-2.0*69.919 + 5.0)",
        "-2.0 + (69.919 + (-907.0 + 7.211))",
    ]

    print(f"📝 Testing {len(test_expressions)} expressions:")

    for i, expr_str in enumerate(test_expressions):
        print(f"\n   Example {i+1}: {expr_str}")

        try:
            # Parse expression
            expr = sympy.parse_expr(expr_str)
            expected_result = float(expr.evalf())

            # Convert to tensors
            target_tensors, valid_mask = expressions_to_tensors(
                [expr],
                depth=config.dag_depth,
                max_digits=config.max_digits,
                max_decimal_places=config.max_decimal_places,
            )

            if not valid_mask[0]:
                print(f"     ❌ Expression not valid for DAG depth {config.dag_depth}")
                continue

            target = target_tensors[0]

            # Create a simple test input using the actual expression
            test_input = tokenize_texts([expr_str], config.block_size, "cpu")

            with torch.no_grad():
                if isinstance(model, PredictorOnlyModel):
                    # DAG Predictor - get predictions
                    pred_digit_logits, pred_V_sign, pred_O, pred_G = model(test_input)

                    # Get the last token position for the complete expression
                    last_token_pos = pred_digit_logits.shape[1] - 1

                    # Extract single prediction tensors for the last token
                    single_digit_logits = pred_digit_logits[
                        0, last_token_pos
                    ]  # (num_initial_nodes, D, base)
                    single_V_sign = pred_V_sign[0, last_token_pos]  # (total_nodes,)
                    single_O = pred_O[0, last_token_pos]  # (dag_depth, total_nodes)
                    single_G = pred_G[0, last_token_pos]  # (dag_depth,)

                    # Get raw (unsharpened) predictions
                    raw_digit_logits = single_digit_logits
                    raw_V_sign = single_V_sign
                    raw_O = single_O
                    raw_G = single_G

                    # Sharpen predictions for cleaner expression display and consistent execution
                    # Note: V_sign, O, and G are already sharpened by STE in the predictor forward pass
                    sharp_digit_logits = _sharpen_digit_predictions(single_digit_logits)
                    sharp_V_sign = single_V_sign  # Already sharpened by STE
                    sharp_O = single_O  # Already sharpened by STE
                    sharp_G = single_G  # Already sharpened by STE

                    # Convert both raw and sharpened tensors to expressions
                    raw_expr = tensor_to_expression(
                        raw_digit_logits,
                        raw_V_sign,
                        raw_O,
                        raw_G,
                        max_digits=config.max_digits,
                        max_decimal_places=config.max_decimal_places,
                    )
                    sharp_expr = tensor_to_expression(
                        sharp_digit_logits,
                        sharp_V_sign,
                        sharp_O,
                        sharp_G,
                        max_digits=config.max_digits,
                        max_decimal_places=config.max_decimal_places,
                    )

                    # Print predicted initial values
                    _print_predicted_initial_values(
                        single_digit_logits, single_V_sign, config
                    )

                    # Print digit probabilities for the first node
                    _print_digit_probabilities(single_digit_logits, node_idx=0)

                    # Try to execute both raw and sharpened DAGs
                    try:
                        executor = DAGExecutor(
                            dag_depth=config.dag_depth,
                            max_digits=config.max_digits,
                            max_decimal_places=config.max_decimal_places,
                        )

                        # Execute both raw and sharpened DAGs
                        raw_result = _execute_dag_prediction(
                            executor, raw_digit_logits, raw_V_sign, raw_O, raw_G
                        )
                        sharp_result = _execute_dag_prediction(
                            executor, sharp_digit_logits, sharp_V_sign, sharp_O, sharp_G
                        )

                        print(f"     --- Predicted DAG ---")
                        print(f"     Input Expression: {expr_str}")
                        print(f"     Raw Expression: {str(raw_expr)}")
                        print(f"     Raw Result: {raw_result:.6f}")
                        print(f"     Sharpened Expression: {str(sharp_expr)}")
                        print(f"     Sharpened Result: {sharp_result:.6f}")

                    except Exception as e:
                        print(f"     - DAG execution failed with exception:")
                        raise
                else:
                    # DAG-GPT model
                    dag_result = model(test_input)
                    _ = dag_result[0, -1].item()  # Take last token position

        except Exception as ex:
            print(f"     ❌ Error processing expression")
            raise


def test_model_behavior(model, config):
    """Test basic model behavior and outputs."""
    print(f"\n🧪 Testing model behavior...")

    model.eval()

    # Create a simple test input using a mathematical expression
    test_input = tokenize_texts(["1.0 + 2.0"], config.block_size, "cpu")

    with torch.no_grad():
        if isinstance(model, PredictorOnlyModel):
            outputs = model(test_input)
            digit_logits, V_sign, O, G = outputs

            print(f"📊 DAG Predictor outputs:")
            print(f"   - Digit logits: {digit_logits.shape}")
            print(f"   - V_sign: {V_sign.shape}")
            print(f"   - O: {O.shape}")
            print(f"   - G: {G.shape}")

            # Get the last token position for the complete expression
            last_token_pos = digit_logits.shape[1] - 1

            # Extract single prediction tensors for the last token
            single_digit_logits = digit_logits[0, last_token_pos]
            single_V_sign = V_sign[0, last_token_pos]
            single_O = O[0, last_token_pos]
            single_G = G[0, last_token_pos]

            # Sharpen predictions
            # Note: V_sign, O, and G are already sharpened by STE in the predictor forward pass
            sharp_digit_logits = _sharpen_digit_predictions(single_digit_logits)
            sharp_V_sign = single_V_sign  # Already sharpened by STE
            sharp_O = single_O  # Already sharpened by STE
            sharp_G = single_G  # Already sharpened by STE

            # Convert to expression
            pred_expr = tensor_to_expression(
                sharp_digit_logits,
                sharp_V_sign,
                sharp_O,
                sharp_G,
                max_digits=config.max_digits,
                max_decimal_places=config.max_decimal_places,
            )

            print(f"   - Predicted Expression: {str(pred_expr)}")

        else:
            dag_results = model(test_input)
            print(f"📊 DAG-GPT outputs:")
            print(f"   - Output shape: {dag_results.shape}")
            print(f"   - Sample values: {dag_results[0, :3].tolist()}")

    print(f"✅ Model behavior test successful")


def main():
    """Main function to test checkpoint on examples."""
    print("🚀 Checkpoint Examples Testing Script")
    print("=" * 50)

    # Check if checkpoint path provided
    if len(sys.argv) < 2:
        print("❌ Error: Checkpoint path is required")
        print("Usage: python scripts/test_checkpoint_examples.py <checkpoint_path>")
        print("")
        print("Examples:")
        print(
            "  python scripts/test_checkpoint_examples.py best_checkpoint/ckpt_predictor_pretrain_best_99.99acc.pt"
        )
        print("  python scripts/test_checkpoint_examples.py wandb://my-run-name")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    print(f"📁 Using checkpoint: {checkpoint_path}")

    # Load checkpoint
    model, config, _ = load_checkpoint(checkpoint_path)

    # Test DAG examples
    test_dag_examples(model, config)

    # Test model behavior
    test_model_behavior(model, config)

    print(f"\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()
