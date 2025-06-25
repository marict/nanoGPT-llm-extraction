import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_checkpoints as cc
from evaluation import comprehensive_evaluate, evaluate_from_dataset_file


class DummyModel:
    def __call__(self, x, targets=None):
        return None, torch.tensor(0.5)

    def eval(self):
        return self

    def train(self):
        return self


class DummyEnc:
    def encode(self, text):
        return [0, 1, 2, 3]  # Longer sequence to avoid issues


def test_evaluate_constant_loss():
    """Test the basic evaluation function with constant loss."""
    path = Path(__file__).resolve().parent / "math_eval.txt"
    model = DummyModel()
    loss = evaluate_from_dataset_file(model, str(path), "cpu")
    assert loss == pytest.approx(0.5)


def test_compare_models_comprehensive_basic():
    """Test the comprehensive comparison with minimal setup."""
    with patch("compare_checkpoints.load_checkpoint") as mock_load, patch(
        "compare_checkpoints.evaluate_from_dataset_file"
    ) as mock_eval:

        # Mock model with basic attributes
        mock_model = MagicMock()
        mock_model.config.dag_depth = 0
        mock_model.get_num_params.return_value = 1000  # Much smaller

        # Mock checkpoint loading
        mock_ckpt = {"model_args": {"dag_depth": 0}}
        mock_load.return_value = (mock_model, mock_ckpt)

        # Mock evaluation
        mock_eval.return_value = 0.5

        dataset_file = Path(__file__).resolve().parent / "math_eval.txt"

        results = cc.compare_models_comprehensive(
            ckpt_baseline="dummy_baseline.pt",
            ckpt_dag="dummy_dag.pt",
            dataset_file=str(dataset_file),
            math_tasks=[],  # Skip math evaluation for speed
            math_max_examples=0,
        )

        # Check that results have the expected structure
        assert "baseline" in results
        assert "dag" in results
        assert "comparison" in results

        # Check that basic metrics are present
        assert "dag_depth" in results["baseline"]
        assert "n_params" in results["baseline"]
        assert "dataset_loss" in results["baseline"]


def test_comprehensive_evaluate_no_data():
    """Test that comprehensive evaluation handles missing data gracefully."""
    with patch("evaluation.evaluate_math") as mock_math_eval:
        mock_math_eval.return_value = {"gsm8k": 0.75, "svamp": 0.80}

        # Mock model
        mock_model = MagicMock()
        mock_model.config.dag_depth = 2

        # Test with non-existent data directory
        results = comprehensive_evaluate(
            mock_model,
            Path("/nonexistent"),
            "cpu",
            eval_iters=10,
            math_tasks=["gsm8k", "svamp"],
            math_max_examples=10,
        )

        # Should still return math evaluation results
        assert "math_eval_gsm8k" in results
        assert "math_eval_svamp" in results
