"""Test that gate accuracy calculation is working correctly."""

import pytest
import torch
import torch.nn.functional as F

from predictor_utils import _compute_g_loss


def test_gate_accuracy_perfect_predictions():
    """Test gate accuracy calculation when predictions are perfect."""

    # Test case 1: All targets are 1 (linear domain), predictions are high
    target_G = torch.tensor([[1.0, 1.0, 1.0]])  # (1, 3) - all linear domain
    pred_logits = torch.tensor(
        [[5.0, 4.0, 6.0]]
    )  # (1, 3) - high logits → sigmoid > 0.5

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Sigmoid of high positive values should be > 0.5, so discrete prediction = 1
    # All targets are 1, so accuracy should be 100%
    assert accuracy.item() == pytest.approx(
        1.0
    ), f"Expected 100% accuracy, got {accuracy.item():.3f}"

    # Test case 2: All targets are 0 (log domain), predictions are low
    target_G = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3) - all log domain
    pred_logits = torch.tensor(
        [[-5.0, -4.0, -6.0]]
    )  # (1, 3) - low logits → sigmoid < 0.5

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Sigmoid of high negative values should be < 0.5, so discrete prediction = 0
    # All targets are 0, so accuracy should be 100%
    assert accuracy.item() == pytest.approx(
        1.0
    ), f"Expected 100% accuracy, got {accuracy.item():.3f}"


def test_gate_accuracy_terrible_predictions():
    """Test gate accuracy calculation when predictions are completely wrong."""

    # Test case 1: All targets are 1, but predictions are low (opposite)
    target_G = torch.tensor([[1.0, 1.0, 1.0]])  # (1, 3) - all linear domain
    pred_logits = torch.tensor(
        [[-5.0, -4.0, -6.0]]
    )  # (1, 3) - low logits → sigmoid < 0.5

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Predictions will be 0 but targets are 1, so accuracy should be 0%
    assert accuracy.item() == pytest.approx(
        0.0
    ), f"Expected 0% accuracy, got {accuracy.item():.3f}"

    # Test case 2: All targets are 0, but predictions are high (opposite)
    target_G = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3) - all log domain
    pred_logits = torch.tensor(
        [[5.0, 4.0, 6.0]]
    )  # (1, 3) - high logits → sigmoid > 0.5

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Predictions will be 1 but targets are 0, so accuracy should be 0%
    assert accuracy.item() == pytest.approx(
        0.0
    ), f"Expected 0% accuracy, got {accuracy.item():.3f}"


def test_gate_accuracy_mixed_predictions():
    """Test gate accuracy calculation with mixed correct/incorrect predictions."""

    # Create a scenario with 3 out of 4 predictions correct
    target_G = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # (1, 4) - mixed targets
    pred_logits = torch.tensor(
        [[5.0, -4.0, 6.0, 2.0]]
    )  # (1, 4) - first 3 correct, last wrong

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Sigmoid outputs: [~1, ~0, ~1, ~0.88]
    # Discrete outputs (>0.5): [1, 0, 1, 1]
    # Targets:               [1, 0, 1, 0]
    # Correct:               [T, T, T, F] → 3/4 = 75%
    expected_accuracy = 3.0 / 4.0
    assert accuracy.item() == pytest.approx(
        expected_accuracy
    ), f"Expected {expected_accuracy:.3f} accuracy, got {accuracy.item():.3f}"


def test_gate_accuracy_edge_cases_around_threshold():
    """Test gate accuracy calculation with predictions right at the 0.5 threshold."""

    # Test with logits that give sigmoid outputs very close to 0.5
    target_G = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # (1, 4)

    # Logits that produce sigmoid values just above and below 0.5
    pred_logits = torch.tensor([[0.1, -0.1, 0.01, -0.01]])  # (1, 4)

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Check the actual sigmoid values
    sigmoid_outputs = torch.sigmoid(pred_logits)
    discrete_preds = (sigmoid_outputs > 0.5).float()

    # Manual calculation
    expected_correct = (discrete_preds == target_G).float()
    expected_accuracy = expected_correct.mean()

    assert accuracy.item() == pytest.approx(expected_accuracy.item()), (
        f"Expected {expected_accuracy.item():.6f} accuracy, got {accuracy.item():.6f}\n"
        f"Sigmoid outputs: {sigmoid_outputs}\n"
        f"Discrete predictions: {discrete_preds}\n"
        f"Targets: {target_G}"
    )


def test_gate_accuracy_batch_dimension():
    """Test gate accuracy calculation across multiple batch samples."""

    # Create a batch with different accuracy patterns
    target_G = torch.tensor(
        [
            [1.0, 0.0, 1.0],  # Sample 1
            [0.0, 1.0, 0.0],  # Sample 2
            [1.0, 1.0, 0.0],  # Sample 3
        ]
    )  # (3, 3)

    pred_logits = torch.tensor(
        [
            [5.0, -5.0, 5.0],  # Sample 1: predictions [1,0,1]
            [-5.0, 5.0, -5.0],  # Sample 2: predictions [0,1,0]
            [5.0, -5.0, 5.0],  # Sample 3: predictions [1,0,1]
        ]
    )  # (3, 3)

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Manual calculation:
    # Sample 1: pred [1,0,1] vs target [1,0,1] → [T,T,T] = 3/3 correct
    # Sample 2: pred [0,1,0] vs target [0,1,0] → [T,T,T] = 3/3 correct
    # Sample 3: pred [1,0,1] vs target [1,1,0] → [T,F,F] = 1/3 correct
    # Total: (3+3+1)/(3+3+3) = 7/9 ≈ 0.778
    expected_accuracy = 7.0 / 9.0

    # Let's also manually verify our calculation
    sigmoid_outputs = torch.sigmoid(pred_logits)
    discrete_preds = (sigmoid_outputs > 0.5).float()
    manual_correct = (discrete_preds == target_G).float()
    manual_accuracy = manual_correct.mean()

    assert accuracy.item() == pytest.approx(expected_accuracy, abs=1e-6), (
        f"Expected {expected_accuracy:.6f} accuracy, got {accuracy.item():.6f}\n"
        f"Manual verification: {manual_accuracy.item():.6f}\n"
        f"Predictions: {discrete_preds}\n"
        f"Targets:     {target_G}\n"
        f"Correct:     {manual_correct}"
    )


def test_gate_accuracy_realistic_scenario():
    """Test a realistic scenario that might explain 80% accuracy from the start."""

    # Scenario: Model is biased towards predicting linear domain (1)
    # If most expressions use addition (linear=1) more than multiplication (log=0),
    # then a model biased towards 1 would get high accuracy early

    # Simulate a batch where 80% of gate targets are 1 (linear domain)
    target_G = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0],  # 4/5 are linear
            [1.0, 1.0, 1.0, 0.0, 0.0],  # 3/5 are linear
            [1.0, 1.0, 1.0, 1.0, 1.0],  # 5/5 are linear
            [1.0, 1.0, 0.0, 0.0, 0.0],  # 2/5 are linear
        ]
    )  # (4, 5) → 14/20 = 70% are linear

    # Model that's biased to predict ~0.8 (sigmoid output) for everything
    # This would discretize to 1 for all predictions
    pred_logits = torch.full((4, 5), 1.386)  # logit that gives sigmoid ≈ 0.8

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # All predictions will be 1 (since sigmoid > 0.5)
    # Accuracy = fraction of targets that are 1 = 14/20 = 70%
    expected_accuracy = 14.0 / 20.0

    assert accuracy.item() == pytest.approx(expected_accuracy, abs=1e-3), (
        f"Expected {expected_accuracy:.3f} accuracy, got {accuracy.item():.3f}\n"
        f"This shows how a model biased toward predicting '1' can achieve high accuracy\n"
        f"if the dataset has more linear operations (addition) than log operations (multiplication)"
    )


def test_gate_accuracy_completely_linear_expressions():
    """Test scenario where all expressions are purely additive (all targets = 1)."""

    # Scenario: All expressions are just additions, so all gates should be 1
    target_G = torch.ones(3, 4)  # (3, 4) - all linear domain

    # Model that predicts moderate positive values (sigmoid ≈ 0.7-0.9)
    pred_logits = torch.tensor(
        [
            [1.0, 1.5, 2.0, 0.8],
            [1.2, 0.9, 1.8, 1.1],
            [1.4, 1.3, 1.6, 0.7],
        ]
    )  # (3, 4)

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # All sigmoid outputs should be > 0.5, so all predictions = 1
    # All targets are 1, so accuracy should be 100%
    assert accuracy.item() == pytest.approx(1.0), (
        f"Expected 100% accuracy for all-linear expressions, got {accuracy.item():.3f}\n"
        f"This scenario could explain high gate accuracy if the dataset\n"
        f"consists mostly of addition-heavy expressions"
    )


def test_gate_accuracy_vs_manual_calculation():
    """Verify that our gate accuracy matches a manual calculation step by step."""

    # Simple test case we can verify by hand
    target_G = torch.tensor([[1.0, 0.0]])  # (1, 2)
    pred_logits = torch.tensor([[2.0, -1.0]])  # (1, 2)

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Manual calculation:
    # 1. Apply sigmoid: sigmoid(2.0) ≈ 0.881, sigmoid(-1.0) ≈ 0.269
    manual_sigmoid = torch.sigmoid(pred_logits)
    expected_sigmoid = torch.tensor([[0.8808, 0.2689]])
    assert torch.allclose(manual_sigmoid, expected_sigmoid, atol=1e-3)

    # 2. Apply threshold: 0.881 > 0.5 → 1, 0.269 < 0.5 → 0
    manual_discrete = (manual_sigmoid > 0.5).float()
    expected_discrete = torch.tensor([[1.0, 0.0]])
    assert torch.equal(manual_discrete, expected_discrete)

    # 3. Compare with targets: [1,0] vs [1,0] → [True, True] → all correct
    manual_correct = (manual_discrete == target_G).float()
    expected_correct = torch.tensor([[1.0, 1.0]])
    assert torch.equal(manual_correct, expected_correct)

    # 4. Calculate accuracy: mean of [1,1] = 1.0
    manual_accuracy = manual_correct.mean()
    assert manual_accuracy.item() == 1.0

    # 5. Verify our function gives the same result
    assert accuracy.item() == pytest.approx(
        manual_accuracy.item()
    ), f"Manual calculation gives {manual_accuracy.item()}, function gives {accuracy.item()}"


def test_gate_accuracy_investigates_80_percent_mystery():
    """Investigate why gate accuracy might hover around 80% from the start."""

    # Hypothesis: If the dataset has an 80/20 split of linear vs log operations,
    # and the model initially predicts randomly around the sigmoid midpoint,
    # it could achieve ~80% accuracy by chance

    # Create a dataset with 80% linear (1) and 20% log (0) operations
    # This simulates a dataset where addition is much more common than multiplication
    target_batch = []
    for _ in range(10):  # 10 samples
        # Each sample has 5 gates, with 80% probability of being linear (1)
        gates = []
        for _ in range(5):
            if torch.rand(1).item() < 0.8:
                gates.append(1.0)  # Linear domain (addition)
            else:
                gates.append(0.0)  # Log domain (multiplication)
        target_batch.append(gates)

    target_G = torch.tensor(target_batch)  # (10, 5)

    # Model that predicts randomly around sigmoid(1.4) ≈ 0.8
    # This simulates an untrained model that's slightly biased toward linear domain
    pred_logits = torch.randn(10, 5) * 0.5 + 1.4  # Mean=1.4, std=0.5

    loss, accuracy = _compute_g_loss(pred_logits, target_G)

    # Print diagnostic information
    sigmoid_outputs = torch.sigmoid(pred_logits)
    discrete_preds = (sigmoid_outputs > 0.5).float()

    # Calculate the actual percentage of 1s in targets
    target_ones_pct = (target_G == 1.0).float().mean().item()

    # Calculate the percentage of 1s in predictions
    pred_ones_pct = (discrete_preds == 1.0).float().mean().item()

    print(f"\nDiagnostic Info:")
    print(f"Target 1s percentage: {target_ones_pct:.1%}")
    print(f"Predicted 1s percentage: {pred_ones_pct:.1%}")
    print(f"Gate accuracy: {accuracy.item():.1%}")
    print(f"Mean sigmoid output: {sigmoid_outputs.mean().item():.3f}")

    # The accuracy should be reasonable (not 0% or 100%) because of the bias
    assert (
        0.1 <= accuracy.item() <= 1.0
    ), f"Accuracy {accuracy.item():.3f} seems unrealistic"

    # If predictions are biased toward 1 and targets are mostly 1, accuracy should be decent
    if target_ones_pct > 0.7 and pred_ones_pct > 0.7:
        assert accuracy.item() > 0.5, (
            f"With {target_ones_pct:.1%} target 1s and {pred_ones_pct:.1%} predicted 1s, "
            f"accuracy {accuracy.item():.1%} seems too low"
        )


def test_gate_accuracy_dataset_bias_hypothesis():
    """Test the hypothesis that 80% gate accuracy comes from dataset bias toward linear operations."""

    # Create a dataset that mimics what might be in the real training data:
    # - Many expressions use addition (linear domain, G=1)
    # - Fewer expressions use multiplication (log domain, G=0)
    # - This is realistic because addition is more common in simple math expressions

    # Simulate 100 samples with 80% being linear operations
    torch.manual_seed(42)  # For reproducible results
    target_G = torch.zeros(20, 5)  # (20, 5) = 100 total gate decisions

    for i in range(20):
        for j in range(5):
            # 80% chance of linear operation (G=1), 20% chance of log operation (G=0)
            if torch.rand(1).item() < 0.8:
                target_G[i, j] = 1.0  # Linear domain (addition)
            else:
                target_G[i, j] = 0.0  # Log domain (multiplication)

    # Count the actual distribution
    actual_linear_pct = (target_G == 1.0).float().mean().item()

    # Scenario 1: Completely untrained model (random predictions around 0.5)
    random_logits = torch.randn(20, 5) * 0.5  # Small random logits around 0
    _, random_accuracy = _compute_g_loss(random_logits, target_G)

    # Scenario 2: Model biased toward linear domain (common early in training)
    # This happens because linear operations might be easier to learn
    biased_logits = torch.randn(20, 5) * 0.3 + 1.5  # Biased toward positive logits
    _, biased_accuracy = _compute_g_loss(biased_logits, target_G)

    # Scenario 3: Well-trained model that mostly gets it right
    # Add some noise to perfect predictions
    perfect_logits = torch.where(
        target_G == 1.0,
        torch.tensor(3.0),  # High positive for linear
        torch.tensor(-3.0),
    )  # High negative for log
    noisy_logits = perfect_logits + torch.randn(20, 5) * 0.5
    _, trained_accuracy = _compute_g_loss(noisy_logits, target_G)

    print(f"\nDataset Analysis:")
    print(f"Actual linear operation percentage: {actual_linear_pct:.1%}")
    print(f"Random model accuracy: {random_accuracy.item():.1%}")
    print(f"Biased model accuracy: {biased_accuracy.item():.1%}")
    print(f"Trained model accuracy: {trained_accuracy.item():.1%}")

    # Key insights:
    # 1. Random model should get ~50% accuracy regardless of dataset bias
    assert (
        0.4 <= random_accuracy.item() <= 0.6
    ), f"Random model should get ~50% accuracy, got {random_accuracy.item():.1%}"

    # 2. Biased model should get accuracy close to the dataset bias percentage
    # If 80% of targets are 1 and model always predicts 1, accuracy = 80%
    expected_biased_accuracy = actual_linear_pct
    assert abs(biased_accuracy.item() - expected_biased_accuracy) < 0.15, (
        f"Biased model accuracy {biased_accuracy.item():.1%} should be close to "
        f"dataset linear percentage {expected_biased_accuracy:.1%}"
    )

    # 3. Well-trained model should achieve high accuracy
    assert (
        trained_accuracy.item() > 0.85
    ), f"Well-trained model should get >85% accuracy, got {trained_accuracy.item():.1%}"

    print(f"\nConclusion:")
    if biased_accuracy.item() >= 0.75:
        print(f"✓ This explains 80% gate accuracy from the start!")
        print(f"  If your dataset has ~80% linear operations (additions),")
        print(f"  then a model biased toward predicting 'linear domain'")
        print(f"  will achieve ~80% accuracy even when untrained.")
    else:
        print(f"✗ Dataset bias alone doesn't explain 80% accuracy.")
        print(f"  Need to investigate other factors.")


def test_gate_accuracy_debugging_checklist():
    """Provide a debugging checklist for investigating gate accuracy issues."""

    # This test always passes but provides diagnostic information

    print(f"\n=== GATE ACCURACY DEBUGGING CHECKLIST ===")
    print(f"")
    print(f"If you're seeing gate accuracy hovering around 80% from the start:")
    print(f"")
    print(f"1. CHECK DATASET BIAS:")
    print(f"   - Count how many target_G values are 1 vs 0 in your training data")
    print(f"   - If ~80% are 1 (linear domain), this explains the 80% accuracy")
    print(
        f"   - Linear operations (addition) might be more common than log operations (multiplication)"
    )
    print(f"")
    print(f"2. CHECK MODEL BIAS:")
    print(f"   - Look at raw prediction logits (before sigmoid)")
    print(
        f"   - If they're consistently positive, model is biased toward linear domain"
    )
    print(f"   - This could be due to initialization or learning dynamics")
    print(f"")
    print(f"3. CHECK SIGMOID OUTPUTS:")
    print(f"   - Print torch.sigmoid(pred_G) to see actual probabilities")
    print(f"   - Values consistently > 0.5 will all discretize to 1")
    print(f"   - Values consistently < 0.5 will all discretize to 0")
    print(f"")
    print(f"4. CHECK LOSS VS ACCURACY RELATIONSHIP:")
    print(f"   - Gate loss decreasing while accuracy stays constant suggests:")
    print(f"     * Model is getting more confident about the same predictions")
    print(f"     * Sigmoid outputs moving further from 0.5 threshold")
    print(f"     * But discrete predictions (accuracy) don't change")
    print(f"")
    print(f"5. VALIDATE CALCULATION:")
    print(f"   - Use this test suite to verify _compute_g_loss is working correctly")
    print(f"   - All tests should pass if the calculation is correct")
    print(f"")

    # Always pass this test
    assert True, "This test provides debugging information"
