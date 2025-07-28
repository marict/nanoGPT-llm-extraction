import pytest
import torch

from models.dag_model import DAGPlanPredictor, GPTConfig


def test_train_uncertainty_params_flag_enabled():
    """Test that uncertainty params have gradients when train_uncertainty_params=True (default)."""
    config = GPTConfig(
        n_embd=64,
        n_head=2,
        n_layer=2,
        dag_depth=2,
        train_uncertainty_params=True,  # Default behavior
    )

    predictor = DAGPlanPredictor(config)

    # Uncertainty params should require gradients
    assert predictor.uncertainty_params.requires_grad
    assert predictor.uncertainty_params.grad_fn is None  # No computations yet

    # Create some dummy input and do a forward pass
    B, T, H = 1, 4, config.n_embd
    hidden = torch.randn(B, T, H, requires_grad=True)

    initial_sgn, digit_probs, operation_probs, statistics = predictor(hidden)

    # Compute a dummy loss that involves uncertainty_params
    dummy_loss = predictor.uncertainty_params.sum()
    dummy_loss.backward()

    # Uncertainty params should have gradients
    assert predictor.uncertainty_params.grad is not None
    assert not torch.allclose(
        predictor.uncertainty_params.grad,
        torch.zeros_like(predictor.uncertainty_params.grad),
    )


def test_train_uncertainty_params_flag_disabled():
    """Test that uncertainty params don't have gradients when train_uncertainty_params=False."""
    config = GPTConfig(
        n_embd=64,
        n_head=2,
        n_layer=2,
        dag_depth=2,
        train_uncertainty_params=False,  # Disable training
    )

    predictor = DAGPlanPredictor(config)

    # Uncertainty params should NOT require gradients
    assert not predictor.uncertainty_params.requires_grad

    # Create some dummy input and do a forward pass
    B, T, H = 1, 4, config.n_embd
    hidden = torch.randn(B, T, H, requires_grad=True)

    initial_sgn, digit_probs, operation_probs, statistics = predictor(hidden)

    # Create a loss that involves both uncertainty_params and other trainable parameters
    # The uncertainty_params won't get gradients since they don't require grad
    dummy_loss = predictor.uncertainty_params.sum() + operation_probs.sum()
    dummy_loss.backward()

    # Uncertainty params should NOT have gradients
    assert predictor.uncertainty_params.grad is None


def test_uncertainty_params_values_stay_zero_when_disabled():
    """Test that uncertainty params remain at their initial zero values when training is disabled."""
    config = GPTConfig(
        n_embd=64, n_head=2, n_layer=2, dag_depth=2, train_uncertainty_params=False
    )

    predictor = DAGPlanPredictor(config)

    # Should be initialized to zeros
    initial_values = predictor.uncertainty_params.clone()
    assert torch.allclose(initial_values, torch.zeros(6))

    # Simulate multiple forward passes that would normally update params
    B, T, H = 1, 4, config.n_embd

    for _ in range(5):
        hidden = torch.randn(B, T, H, requires_grad=True)
        initial_sgn, digit_probs, operation_probs, statistics = predictor(hidden)

        # Create a dummy loss involving the uncertainty params
        dummy_loss = predictor.uncertainty_params.sum() + operation_probs.sum()
        dummy_loss.backward()

        # Simulate an optimizer step (but uncertainty_params should not change)
        with torch.no_grad():
            for param in predictor.parameters():
                if param.requires_grad and param.grad is not None:
                    param -= 0.01 * param.grad
                param.grad = None

    # Uncertainty params should still be zeros
    final_values = predictor.uncertainty_params.clone()
    assert torch.allclose(final_values, torch.zeros(6))
    assert torch.allclose(initial_values, final_values)


def test_default_config_has_training_enabled():
    """Test that the default GPTConfig has uncertainty training enabled."""
    config = GPTConfig()
    assert config.train_uncertainty_params is True


if __name__ == "__main__":
    test_train_uncertainty_params_flag_enabled()
    test_train_uncertainty_params_flag_disabled()
    test_uncertainty_params_values_stay_zero_when_disabled()
    test_default_config_has_training_enabled()
    print("All tests passed!")
