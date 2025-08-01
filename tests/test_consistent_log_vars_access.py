"""
Test for consistent uncertainty_params access pattern across model types.

This test ensures that the strange logging behavior where uncertainty_params would stop
being logged at different steps (step 7 for stats, step 80 for others) is fixed
by verifying that both GPT and PredictorOnlyModel provide consistent access.
"""

import pytest
import torch
from dag_logger import DAGLogger

from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


class TestConsistentUncertaintyParamsAccess:
    """Test consistent access to uncertainty_params across different model types."""

    def test_gpt_model_dag_predictor_access(self):
        """Test that GPT model provides consistent dag_predictor access."""
        config = GPTConfig(
            dag_depth=4,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        model = GPT(config)

        # Test that dag_predictor property works
        assert model.dag_predictor is not None
        assert hasattr(model.dag_predictor, "uncertainty_params")
        assert model.dag_predictor.uncertainty_params.shape == torch.Size([6])

        # Test that we can access uncertainty_params directly
        uncertainty_params = model.dag_predictor.uncertainty_params
        assert uncertainty_params.dtype == torch.float32

    def test_predictor_only_model_dag_predictor_access(self):
        """Test that PredictorOnlyModel maintains dag_predictor access."""
        config = PredictorOnlyConfig(
            dag_depth=4,
            n_layer=1,
            n_embd=32,
            n_head=4,
            vocab_size=50304,
            block_size=128,
        )
        model = PredictorOnlyModel(config)

        # Test that dag_predictor access works
        assert model.dag_predictor is not None
        assert hasattr(model.dag_predictor, "uncertainty_params")
        assert model.dag_predictor.uncertainty_params.shape == torch.Size([6])

        # Test that we can access uncertainty_params directly
        uncertainty_params = model.dag_predictor.uncertainty_params
        assert uncertainty_params.dtype == torch.float32

    def test_gpt_model_dag_depth_zero_returns_none(self):
        """Test that GPT model with dag_depth=0 returns None for dag_predictor."""
        config = GPTConfig(
            dag_depth=0,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        model = GPT(config)

        # Should return None for dag_depth=0
        assert model.dag_predictor is None

    def test_consistent_access_pattern_both_models(self):
        """Test that both models use the same access pattern without conditionals."""
        # GPT model
        gpt_config = GPTConfig(
            dag_depth=3,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        gpt_model = GPT(gpt_config)

        # PredictorOnly model
        pred_config = PredictorOnlyConfig(
            dag_depth=3,
            n_layer=1,
            n_embd=32,
            n_head=4,
            vocab_size=50304,
            block_size=128,
        )
        pred_model = PredictorOnlyModel(pred_config)

        # Both should work with identical access pattern
        def get_uncertainty_params(model):
            return model.dag_predictor.uncertainty_params

        gpt_uncertainty_params = get_uncertainty_params(gpt_model)
        pred_uncertainty_params = get_uncertainty_params(pred_model)

        # Both should have same shape and be accessible
        assert (
            gpt_uncertainty_params.shape
            == pred_uncertainty_params.shape
            == torch.Size([6])
        )

    def test_dag_logger_includes_uncertainty_params(self):
        """Test that DAGLogger now includes uncertainty_params in its logging."""
        config = GPTConfig(
            dag_depth=4,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        model = GPT(config)

        # Simulate forward pass to populate model state
        x = torch.randint(0, config.vocab_size, (2, 16))
        model(x)

        # Create DAGLogger and get logging dict
        logger = DAGLogger()
        logger.compute_log_statistics(model)
        log_dict = logger.get_wandb_logging_dict(model)

        # Verify uncertainty_params are included (weights are redundant since weights = exp(-params))
        expected_uncertainty_params_keys = [
            "uncertainty_params/sign",
            "uncertainty_params/digit",
            "uncertainty_params/op",
            "uncertainty_params/value",
            "uncertainty_params/exec",
            "uncertainty_params/stats",
        ]

        for key in expected_uncertainty_params_keys:
            assert key in log_dict, f"Missing {key} in DAGLogger output"

        # Verify no redundant weights keys are logged
        weights_keys = [k for k in log_dict.keys() if k.startswith("weights/")]
        assert len(weights_keys) == 0, f"Found redundant weights keys: {weights_keys}"

    def test_no_backwards_compatibility_shims(self):
        """Test that we removed old access patterns without backwards compatibility."""
        config = GPTConfig(
            dag_depth=4,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        model = GPT(config)

        # The old conditional access pattern should NOT exist anywhere
        # We should only have the consistent dag_predictor access

        # Test that the new consistent access works
        assert model.dag_predictor.uncertainty_params is not None

        # Verify there are no old-style access methods lingering around
        # (This is more of a code structure test)
        assert hasattr(model, "dag_predictor"), "Should have dag_predictor property"

        # For GPT models, verify the property correctly routes to the internal structure
        assert model.dag_predictor is model.dag.plan_predictor

    def test_logging_simulation_multiple_steps(self):
        """Simulate the logging scenario that was previously failing."""
        config = GPTConfig(
            dag_depth=4,
            n_layer=1,
            n_head=4,
            n_embd=32,
            vocab_size=50304,
            block_size=128,
        )
        model = GPT(config)

        # Simulate multiple training steps where uncertainty_params access was failing
        for step in [1, 7, 50, 80, 100]:  # These were the problematic steps
            # Simulate forward pass
            x = torch.randint(0, config.vocab_size, (2, 16))
            model(x)

            # This access pattern should work consistently at ALL steps
            try:
                uncertainty_params = model.dag_predictor.uncertainty_params
                uncertainty_weights = torch.exp(-uncertainty_params)

                # Verify we can convert to logging format
                log_dict = {
                    "uncertainty_params/sign": uncertainty_params[0].item(),
                    "uncertainty_params/digit": uncertainty_params[1].item(),
                    "uncertainty_params/op": uncertainty_params[2].item(),
                    "uncertainty_params/value": uncertainty_params[3].item(),
                    "uncertainty_params/exec": uncertainty_params[4].item(),
                    "uncertainty_params/stats": uncertainty_params[5].item(),
                    "uncertainty_weights/sign": uncertainty_weights[0].item(),
                    "uncertainty_weights/digit": uncertainty_weights[1].item(),
                    "uncertainty_weights/op": uncertainty_weights[2].item(),
                    "uncertainty_weights/value": uncertainty_weights[3].item(),
                    "uncertainty_weights/exec": uncertainty_weights[4].item(),
                    "uncertainty_weights/stats": uncertainty_weights[5].item(),
                }

                # All values should be finite
                for key, value in log_dict.items():
                    assert torch.isfinite(
                        torch.tensor(value)
                    ), f"Non-finite value at step {step}: {key}={value}"

            except AttributeError as e:
                pytest.fail(f"uncertainty_params access failed at step {step}: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected error at step {step}: {e}")
