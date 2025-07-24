import random

import pytest
import sympy
import torch

from data.dagset.expression_to_string import (
    convert_number_to_english,
    format_expression_string,
    number_to_string,
)
from data.dagset.streaming import (
    DAGStructureDataset,
    _apply_sympy_op,
    generate_single_dag_example,
    generate_uniform_digit_number,
    plan_to_tensors,
)
from models.dag_model import OP_NAMES


# -----------------------------------------------------------------------------
# Simple number/formatting helpers
# -----------------------------------------------------------------------------
def test_convert_number_to_english_basic_cases():
    cases = [
        (0, "zero"),
        (5, "five"),
        (-5, "negative five"),
        (3.14, "three point one four"),
    ]
    for value, expected in cases:
        result = convert_number_to_english(value, max_decimal_places=2)
        assert result == expected


def test_number_to_string_integer_handling():
    rng = random.Random(42)
    # With probability 1.0 we should drop the ".0" suffix for integers
    assert number_to_string(10, rng, integer_no_decimal_probability=1.0) == "10"

    rng = random.Random(42)
    # With probability 0.0 we keep the ".0" suffix
    assert number_to_string(10, rng, integer_no_decimal_probability=0.0) == "10.0"


# -----------------------------------------------------------------------------
# Expression formatting & permutation helpers
# -----------------------------------------------------------------------------
def test_format_expression_string_conversion_and_identity():
    # Create an expression without evaluation
    a = sympy.Symbol("2.0")
    b = sympy.Symbol("2.0")
    expr = sympy.Add(a, b, evaluate=False)
    initial_values = [2.0, 2.0]

    # No conversion → expression should have basic spacing
    no_conv, style = format_expression_string(
        expr, english_conversion_probability=0.0, seed=1
    )
    assert "2.0 + 2.0" in no_conv

    # Note: The actual English conversion of numbers is now done in _generate_expression
    # So this test only checks if operation conversion still works
    conv, style = format_expression_string(
        expr, english_conversion_probability=1.0, seed=1
    )
    # At least one operator word should appear ("plus" or "added to")
    assert any(op in conv for op in ["plus", "added to"])


# -----------------------------------------------------------------------------
# SymPy operation helper
# -----------------------------------------------------------------------------
def test_apply_sympy_op_correctness():
    a, b = sympy.Integer(3), sympy.Integer(2)
    expected_values = {
        "add": 5,
        "subtract": 1,
        "multiply": 6,
        "divide": sympy.Rational(3, 2),
        "identity": 3,  # Identity operation returns the first operand
    }
    for op_name, expected in expected_values.items():
        expr = _apply_sympy_op(op_name, a, b)
        assert sympy.simplify(expr - expected) == 0

    # Unknown operation should raise error
    with pytest.raises(ValueError):
        _apply_sympy_op("unknown", a, b)


# -----------------------------------------------------------------------------
# Tensor helpers & dataset generation
# -----------------------------------------------------------------------------


def test_float_to_digit_onehot_shape_and_properties():
    from data.dagset.streaming import (  # local import to avoid circular issues
        float_to_digit_onehot,
    )

    tensor = float_to_digit_onehot(12.34, max_digits=2, max_decimal_places=2)
    # Expected shape: D = max_digits + max_decimal_places = 4
    assert tensor.shape == (4, 10)
    # Each row should be a valid one-hot (sum equals 1)
    assert torch.allclose(tensor.sum(dim=1), torch.ones(4))


def test_float_to_digit_onehot_base22_valid_indices():
    """Test that float_to_digit_onehot produces valid indices for base 22."""
    from data.dagset.streaming import (
        float_to_digit_onehot,
        generate_uniform_digit_number,
    )

    base = 22
    max_digits = 3
    max_decimal_places = 3

    # Test various values that could be generated in base 22
    test_values = [
        1.0,
        21.0,
        21.99,
        485.0,  # Edge cases near base limits
        # Generate some random values using the same pipeline as training
        generate_uniform_digit_number(
            seed=42,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
        ),
        generate_uniform_digit_number(
            seed=123,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
        ),
        generate_uniform_digit_number(
            seed=456,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
        ),
    ]

    for value in test_values:
        tensor = float_to_digit_onehot(value, max_digits, max_decimal_places, base)

        # Check shape
        expected_shape = (max_digits + max_decimal_places, base)
        assert tensor.shape == expected_shape, f"Wrong shape for value {value}"

        # Check that all rows are valid one-hot
        assert torch.allclose(
            tensor.sum(dim=1), torch.ones(tensor.shape[0])
        ), f"Not valid one-hot for value {value}"

        # Check that all indices are within valid range [0, base-1]
        indices = torch.argmax(tensor, dim=1)
        max_index = torch.max(indices).item()
        min_index = torch.min(indices).item()

        assert min_index >= 0, f"Negative index {min_index} for value {value}"
        assert max_index < base, f"Index {max_index} >= base {base} for value {value}"

        # Also check that indices are actually integers in valid range
        for i, idx in enumerate(indices):
            assert (
                0 <= idx.item() < base
            ), f"Invalid index {idx.item()} at position {i} for value {value} (base {base})"


def test_plan_to_tensors_base22_pipeline():
    """Test the full plan_to_tensors pipeline with base 22 to catch invalid indices."""
    from data.dagset.streaming import (
        generate_uniform_digit_number,
        plan_to_tensors,
    )

    base = 22
    max_digits = 3
    max_decimal_places = 3
    depth = 4

    # Generate initial values using the same pipeline as training
    initial_values = []
    for i in range(depth + 1):  # depth + 1 initial values
        value = generate_uniform_digit_number(
            seed=42 + i,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
            allow_zero=False,
        )
        initial_values.append(value)

    operations = ["add", "multiply", "subtract", "identity"]

    # Convert to tensors using plan_to_tensors (this is what training uses)
    structure_dict = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=base,
        depth=depth,
        train=False,  # Use full mode to get all tensors
    )

    # Check the digit tensor that gets used for targets
    digits_tensor = structure_dict["initial_digits"]  # Shape: (nodes, D, base)

    # For each node and each digit position, check that argmax gives valid indices
    for node_idx in range(digits_tensor.shape[0]):
        for digit_pos in range(digits_tensor.shape[1]):
            digit_onehot = digits_tensor[node_idx, digit_pos]  # Shape: (base,)

            # Check that it's a valid one-hot
            assert torch.allclose(
                digit_onehot.sum(), torch.tensor(1.0)
            ), f"Invalid one-hot at node {node_idx}, digit {digit_pos}"

            # Check that argmax gives valid index
            index = torch.argmax(digit_onehot).item()
            assert (
                0 <= index < base
            ), f"Invalid index {index} at node {node_idx}, digit {digit_pos} (base {base})"

    # Also test what happens when we flatten for loss computation (like in training)
    B, T = 1, 1  # Simulate batch and sequence dimensions
    target_digits = digits_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
    target_flat = target_digits.reshape(-1, base)  # Flatten like in _compute_digit_loss
    target_idx = target_flat.argmax(dim=-1)  # Convert to indices like in training

    # Check that all flattened indices are valid
    max_target_idx = torch.max(target_idx).item()
    min_target_idx = torch.min(target_idx).item()

    assert min_target_idx >= 0, f"Negative target index: {min_target_idx}"
    assert max_target_idx < base, f"Target index {max_target_idx} >= base {base}"

    print(f"✅ Base {base} pipeline test passed!")
    print(f"   Generated values: {[round(v, 3) for v in initial_values]}")
    print(f"   Target indices range: [{min_target_idx}, {max_target_idx}]")


def test_base_config_flow_train_predictor():
    """Test that base configuration flows correctly from config to model and data."""
    import sys

    sys.path.append(".")

    from checkpoint_manager import CheckpointManager
    from predictor_config import DAGTrainConfig
    from training_utils import load_config_file, update_config

    # Load the actual config file that uses base=22
    cfg = DAGTrainConfig()
    config_dict = load_config_file("config/train_predictor_config.py")
    update_config(cfg, config_dict)

    # Verify config has base=22
    assert cfg.base == 22, f"Config should have base=22, got {cfg.base}"

    # Test model configuration creation (simulate checkpoint manager)
    checkpoint_manager = CheckpointManager("dag")
    model, model_config = checkpoint_manager.initialize_dag_model(
        cfg, checkpoint=None, device="cpu"
    )

    # Check that model config got the correct base
    assert (
        model_config.base == 22
    ), f"Model config should have base=22, got {model_config.base}"

    # Check that model's digit predictor has correct output dimensions
    if hasattr(model, "dag_predictor"):
        # PredictorOnly model
        predictor = model.dag_predictor
    else:
        # Full GPT model with DAG
        predictor = model.dag.plan_predictor

    # Check the digit predictor output dimensions
    # It should have shape: (num_scratch_nodes * (1 + digits_per_number * base))
    num_scratch_nodes = cfg.dag_depth + 1  # 5 nodes for depth 4
    digits_per_number = cfg.max_digits + cfg.max_decimal_places  # 6 digits total
    expected_output_dim = num_scratch_nodes * (1 + digits_per_number * cfg.base)

    actual_output_dim = predictor.initial_values_predictor[-1].out_features
    assert (
        actual_output_dim == expected_output_dim
    ), f"Model predictor output dim should be {expected_output_dim}, got {actual_output_dim}"

    print(f"✅ Base config flow test passed!")
    print(f"   Config base: {cfg.base}")
    print(f"   Model config base: {model_config.base}")
    print(
        f"   Model predictor output dim: {actual_output_dim} (expected: {expected_output_dim})"
    )


def test_real_training_dataloader_base22():
    """Test that the actual training dataloader produces valid base 22 indices."""
    import sys

    sys.path.append(".")

    from data.dagset.streaming import create_dag_structure_dataloaders
    from predictor_config import DAGTrainConfig
    from training_utils import load_config_file, update_config

    # Load the actual config file that uses base=22
    cfg = DAGTrainConfig()
    config_dict = load_config_file("config/train_predictor_config.py")
    update_config(cfg, config_dict)

    # Create dataloader exactly like train_predictor.py does
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=cfg.batch_size,
        val_batch_size=cfg.batch_size,
        max_depth=cfg.dag_depth,
        seed=cfg.seed,
        english_conversion_probability=cfg.english_conversion_probability,
        integer_no_decimal_probability=cfg.integer_no_decimal_probability,
        expression_simplification_probability=cfg.expression_simplification_probability,
        expression_expansion_probability=cfg.expression_expansion_probability,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        base=cfg.base,
        allowed_operations=None,  # Use all operations
        printing_style_probs=cfg.printing_style_probs,
    )

    # Generate a few batches and check all target indices
    print(f"Testing real dataloader with config base={cfg.base}")

    for batch_num in range(3):  # Check 3 batches
        texts, structures, examples = next(train_loader)

        target_digits = structures["initial_digits"]  # Shape: (B, N, D, base)
        batch_size, num_nodes, num_digit_positions, base = target_digits.shape

        # Verify shape matches config
        assert (
            base == cfg.base
        ), f"Batch {batch_num}: target tensor base {base} != config base {cfg.base}"

        # Flatten and get indices like training does
        target_flat = target_digits.reshape(-1, base)
        target_idx = target_flat.argmax(dim=-1)

        # Check all indices are valid
        invalid_indices = target_idx[target_idx >= cfg.base]
        if len(invalid_indices) > 0:
            print(
                f"❌ Batch {batch_num}: Found {len(invalid_indices)} invalid indices!"
            )
            print(
                f"   Invalid indices: {invalid_indices[:10].tolist()}"
            )  # Show first 10
            print(f"   Config base: {cfg.base}")
            print(f"   Target tensor shape: {target_digits.shape}")
            assert False, f"Invalid indices found in batch {batch_num}"

        min_idx = torch.min(target_idx).item()
        max_idx = torch.max(target_idx).item()
        print(
            f"   Batch {batch_num}: indices range [{min_idx}, {max_idx}] (base {cfg.base}) ✅"
        )

    print(f"✅ Real training dataloader test passed!")


def test_float_to_digit_onehot_base10_edge_cases():
    """Test float_to_digit_onehot with base 10 and problematic floating point values."""
    from data.dagset.streaming import float_to_digit_onehot

    base = 10
    max_digits = 4
    max_decimal_places = 6

    # Test values that might cause floating point precision issues
    problematic_values = [
        9.999999999,  # Near integer boundary
        99.99999999,  # Near larger integer boundary
        0.999999999,  # Near fractional boundary
        123.999999,  # Mixed integer/fractional near boundary
        999.999999,  # Large value near boundary
        9.9999,  # Shorter precision issue
        # Values that when multiplied by 10 repeatedly might exceed 9
        0.9999999999,
        0.8999999999,
        # Large values that might cause overflow in digit conversion
        9999.9999,
    ]

    print(f"Testing base {base} with problematic floating point values...")

    for i, value in enumerate(problematic_values):
        print(f"\nTesting value {i}: {value}")

        # Check the intermediate conversion steps manually
        abs_val = abs(value)
        int_part = int(abs_val)
        frac_part = abs_val - int_part

        print(f"  int_part: {int_part}, frac_part: {frac_part}")

        # Check fractional digit conversion (where the bug likely is)
        frac_digits = []
        temp_frac = frac_part
        for j in range(max_decimal_places):
            temp_frac_times_base = temp_frac * base
            digit = int(temp_frac_times_base)

            print(
                f"    Step {j}: temp_frac={temp_frac:.12f}, temp_frac*{base}={temp_frac_times_base:.12f}, digit={digit}"
            )

            if digit >= base:
                print(f"    *** ERROR: digit {digit} >= base {base} at step {j} ***")
                print(f"    *** This will cause invalid target indices! ***")

            frac_digits.append(digit)
            temp_frac = temp_frac_times_base - digit

        # Now test the actual function
        tensor = float_to_digit_onehot(value, max_digits, max_decimal_places, base)
        indices = torch.argmax(tensor, dim=1)
        max_index = torch.max(indices).item()

        print(f"  Final indices: {indices.tolist()}")
        print(f"  Max index: {max_index}")

        if max_index >= base:
            print(f"  *** FUNCTION ERROR: Max index {max_index} >= base {base} ***")
            assert (
                False
            ), f"float_to_digit_onehot produced invalid index {max_index} for base {base} with value {value}"


def test_model_logits_shape_base_mismatch():
    """Test if model outputs wrong logit shapes that could cause invalid indices."""
    import sys

    sys.path.append(".")

    from checkpoint_manager import CheckpointManager
    from predictor_config import DAGTrainConfig
    from predictor_utils import tokenize_texts
    from training_utils import load_config_file, update_config

    # Load the actual config file
    cfg = DAGTrainConfig()
    config_dict = load_config_file(
        "config/train_predictor_default.py"
    )  # Use default base=10 config
    update_config(cfg, config_dict)

    print(
        f"Testing with config base={cfg.base}, max_digits={cfg.max_digits}, max_decimal_places={cfg.max_decimal_places}"
    )

    # Create model
    checkpoint_manager = CheckpointManager("dag")
    model, model_config = checkpoint_manager.initialize_dag_model(
        cfg, checkpoint=None, device="cpu"
    )
    model.eval()

    # Create dummy input
    texts = ["2 + 3 * 4"]
    input_tokens = tokenize_texts(texts, cfg.block_size, "cpu")

    print(f"Input tokens shape: {input_tokens.shape}")

    # Forward pass
    with torch.no_grad():
        if hasattr(model, "dag_predictor"):
            # PredictorOnly model
            pred_sgn, _, pred_ops = model(input_tokens)
            predictor = model.dag_predictor
        else:
            # Full GPT model
            hidden = model.forward_hidden(input_tokens)
            pred_sgn, _, pred_ops = model.dag.plan_predictor(hidden)
            predictor = model.dag.plan_predictor

        # Check digit logits
        if hasattr(predictor, "last_digit_logits"):
            digit_logits = predictor.last_digit_logits  # Should be (B, T, N, D, base)
            print(f"Digit logits shape: {digit_logits.shape}")

            expected_base = cfg.base
            actual_base = digit_logits.shape[-1]

            print(f"Expected base: {expected_base}")
            print(f"Actual base dimension: {actual_base}")

            if actual_base != expected_base:
                print(
                    f"❌ BASE MISMATCH: Model outputs base {actual_base} but config expects {expected_base}"
                )
                print(
                    f"This could cause invalid target indices if targets are generated for base {expected_base}"
                )
                print(f"but model predicts logits for base {actual_base}")

                # This could be the bug! If targets are base 10 (indices 0-9) but
                # model predicts base 22 logits, argmax could give indices 0-21
                assert False, f"Model base mismatch: {actual_base} != {expected_base}"

            # Test a forward pass and see what indices we get
            # Flatten like the training code does
            B, T, N, D, base = digit_logits.shape
            logits_flat = digit_logits.reshape(-1, base)

            # Get predicted indices
            pred_indices = torch.argmax(logits_flat, dim=-1)
            max_pred_idx = torch.max(pred_indices).item()
            min_pred_idx = torch.min(pred_indices).item()

            print(f"Model predicted indices range: [{min_pred_idx}, {max_pred_idx}]")
            print(f"Expected range for base {cfg.base}: [0, {cfg.base-1}]")

            if max_pred_idx >= cfg.base:
                print(
                    f"❌ MODEL PREDICTION ERROR: Model predicted index {max_pred_idx} >= base {cfg.base}"
                )
                # This would be really bad - model is predicting invalid indices
        else:
            print("❌ No last_digit_logits found in predictor")

    print("✅ Model logits shape test completed")


def test_plan_to_tensors_shapes():
    init_vals = [1.2, -3.4]
    operations = ["add"]
    structure_dict = plan_to_tensors(
        init_vals, operations, max_digits=2, max_decimal_places=2
    )

    # Check that we get the expected structure dict keys
    expected_keys = {
        "initial_sgn",
        "initial_log",
        "initial_digits",
        "operation_probs",
        "depth",
        "operations",
        "final_value_exec",
    }
    assert set(structure_dict.keys()) == expected_keys

    # Check tensor shapes - with depth=1 and 2 nodes, we get 2 nodes total
    assert structure_dict["initial_sgn"].shape == (2,)  # depth + 1 = 2 nodes
    assert structure_dict["initial_digits"].shape == (
        2,
        4,
        10,
    )  # (nodes, digits, classes)
    assert structure_dict["operation_probs"].shape == (1, len(OP_NAMES))  # (depth, ops)

    # Check signs match original values
    assert torch.equal(structure_dict["initial_sgn"][:2], torch.tensor([1.0, -1.0]))
    # Operation one-hot should have exactly one "1" at the correct index
    assert structure_dict["operation_probs"].sum() == 1
    assert structure_dict["operation_probs"][0, OP_NAMES.index("add")].item() == 1


def test_generate_single_dag_example_basic_properties():
    example = generate_single_dag_example(
        depth=3, seed=999, max_digits=2, max_decimal_places=2
    )
    assert example.depth == 3
    assert len(example.initial_values) == 4  # depth + 1
    # Operations tensor first dimension should equal depth
    assert example.operations.shape[0] == example.depth


def test_dag_structure_dataset_batch_shapes():
    dataset = DAGStructureDataset(
        max_depth=2, seed=321, max_digits=2, max_decimal_places=2
    )
    texts, batched_struct, _ = dataset.generate_batch(batch_size=2)

    # Basic sanity checks
    assert len(texts) == 2
    assert batched_struct["initial_sgn"].shape == (2, 3)  # batch, nodes(depth+1)
    assert batched_struct["operation_probs"].shape == (2, 2, len(OP_NAMES))


def test_batched_target_values_consistency():
    """Test that batched target values match the values from examples."""
    from data.dagset.streaming import create_dag_structure_dataloaders

    # Create a small test dataset - use validation loader for this test
    # since DAGTrainExample doesn't store these values for performance
    _, val_loader = create_dag_structure_dataloaders(
        train_batch_size=3,
        val_batch_size=3,
        max_depth=2,
        seed=42,
        max_digits=3,
        max_decimal_places=2,
    )

    # Get a batch from validation loader (has full attributes)
    texts, structures, examples = next(iter(val_loader))

    # Verify that batched target values match example values
    batched_initial_values = structures["target_initial_values"]
    batched_final_exec = structures["target_final_exec"]

    for i, example in enumerate(examples):
        # Skip training examples since they don't have these attributes for performance
        from data.dagset.streaming import DAGTrainExample

        if isinstance(example, DAGTrainExample):
            continue

        # Check initial values
        num_values = min(len(example.initial_values), batched_initial_values.size(1))
        for j in range(num_values):
            expected = example.initial_values[j]
            actual = batched_initial_values[i, j].item()
            assert (
                abs(expected - actual)
                < 1e-4  # Slightly increased tolerance for float precision
            ), f"Initial value mismatch at [{i},{j}]: {expected} vs {actual}"

        # Check final exec value
        if example.final_value_exec is not None:
            expected_final = example.final_value_exec
            actual_final = batched_final_exec[i].item()
            assert (
                abs(expected_final - actual_final) < 1e-5
            ), f"Final exec value mismatch at [{i}]: {expected_final} vs {actual_final}"


def test_generate_uniform_digit_number_zero_handling():
    """Test that zero is never generated when allow_zero=False."""
    # Generate a large number of values with allow_zero=False
    # and verify that none of them are zero
    for _ in range(100):  # Test 100 random seeds
        seed = random.randint(0, 10000)
        result = generate_uniform_digit_number(
            seed=seed,
            max_digits=1,
            max_decimal_places=0,
            allow_zero=False,
            integer_no_decimal_probability=0.0,
        )
        assert result != 0
        assert result != 0.0

        # Also test with integer_no_decimal_probability=1.0
        result = generate_uniform_digit_number(
            seed=seed,
            max_digits=1,
            max_decimal_places=0,
            allow_zero=False,
            integer_no_decimal_probability=1.0,
        )
        assert result != 0  # This should catch both int(0) and float(0.0)


def test_generate_uniform_digit_number_integer_conversion():
    """Test that integer_no_decimal_probability controls integer type conversion."""
    # Test with integer_no_decimal_probability=1.0
    # All whole numbers should be returned as integers
    integer_count = 0
    total_trials = 50

    for _ in range(total_trials):
        seed = random.randint(0, 10000)
        result = generate_uniform_digit_number(
            seed=seed,
            max_digits=1,
            max_decimal_places=0,  # Force whole numbers
            allow_zero=True,
            integer_no_decimal_probability=1.0,
        )
        if isinstance(result, int):
            integer_count += 1

    # All results should be integers
    assert (
        integer_count == total_trials
    ), f"Expected all {total_trials} results to be integers, but got {integer_count}"

    # Test with integer_no_decimal_probability=0.0
    # All whole numbers should be returned as floats
    float_count = 0
    for _ in range(total_trials):
        seed = random.randint(0, 10000)
        result = generate_uniform_digit_number(
            seed=seed,
            max_digits=1,
            max_decimal_places=0,  # Force whole numbers
            allow_zero=True,
            integer_no_decimal_probability=0.0,
        )
        if isinstance(result, float):
            float_count += 1

    # All results should be floats
    assert (
        float_count == total_trials
    ), f"Expected all {total_trials} results to be floats, but got {float_count}"
