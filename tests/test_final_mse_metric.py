import math
from contextlib import nullcontext

import pytest
import torch

from data.dagset.streaming import generate_single_dag_example
from predictor_config import DAGTrainConfig
from predictor_utils import evaluate_dag_model


class DummyPredictor(torch.nn.Module):
    """A minimal stub model that returns pre-baked predictions.

    It mimics the interface expected by `evaluate_dag_model` when
    `cfg.full_backbone == False` (stand-alone predictor)."""

    def __init__(self, pred_sgn, pred_digits, pred_ops):
        super().__init__()
        # Store tensors as parameters so they move with .to(device) if needed
        self.register_buffer("_pred_sgn", pred_sgn)
        self.register_buffer("_pred_ops", pred_ops)
        # The evaluation routine expects `model.dag_predictor.digit_logits`
        dummy = torch.nn.Module()
        dummy.register_buffer("digit_logits", pred_digits)
        self.dag_predictor = dummy

    def forward(self, _):  # noqa: D401
        # Return sign logits/probs, unused placeholder, and op probs
        return self._pred_sgn, None, self._pred_ops


@pytest.fixture(scope="module")
def example_structures():
    # Generate a single deterministic DAG example
    depth = 2
    example = generate_single_dag_example(
        depth=depth, seed=123, max_digits=2, max_decimal_places=2
    )

    structures = {
        "initial_sgn": example.signs.unsqueeze(0),  # (B=1, N)
        "initial_log": example.log_magnitudes.unsqueeze(0),  # (B=1, N)
        "initial_digits": example.digits.unsqueeze(0),  # (B=1, N, D, 10)
        "operation_probs": example.operations.unsqueeze(0),  # (B=1, depth, n_ops)
    }

    texts = [example.text]
    examples = [example]
    return texts, structures, examples


def _run_eval(texts, structures, examples, predictor):
    # Minimal config to satisfy evaluate_dag_model
    cfg = DAGTrainConfig()
    cfg.full_backbone = False
    cfg.max_digits = 2
    cfg.max_decimal_places = 2
    cfg.sequence_length = 16
    cfg.eval_iters = 1

    device = "cpu"
    ctx = nullcontext()

    # Single-batch val_loader
    val_loader = iter([(texts, structures, examples)])

    metrics = evaluate_dag_model(
        predictor, val_loader, device, ctx, cfg, eval_iters=1, seed=42
    )
    return metrics


def test_final_mse_perfect_predictions(example_structures):
    texts, structures, examples = example_structures

    # Prepare perfect predictions (copy targets)
    tgt_sgn = structures["initial_sgn"].unsqueeze(1)  # (B,1,N)
    tgt_digits = structures["initial_digits"].unsqueeze(1)  # (B,1,N,D,10)
    tgt_ops = structures["operation_probs"].unsqueeze(1)  # (B,1,depth,n_ops)

    # Use confident logits for perfect prediction
    tgt_digits_logits = tgt_digits.clone() * 50.0
    model = DummyPredictor(tgt_sgn.clone(), tgt_digits_logits.clone(), tgt_ops.clone())

    metrics = _run_eval(texts, structures, examples, model)

    assert metrics["final_mse"] < 1e-3  # Should be nearly zero for perfect predictions


def test_final_mse_incorrect_predictions(example_structures):
    texts, structures, examples = example_structures

    # Prepare incorrect predictions by flipping the sign of first initial value
    tgt_sgn = structures["initial_sgn"].clone()
    tgt_sgn[0, 0] *= -1  # invert sign of the first value
    pred_sgn = tgt_sgn.unsqueeze(1)  # (B,1,N)

    pred_digits = structures["initial_digits"].unsqueeze(1).clone()
    # Convert one-hot probs to confident logits
    pred_digits_logits = pred_digits * 50.0  # set correct digit high
    pred_ops = structures["operation_probs"].unsqueeze(1).clone()

    model = DummyPredictor(pred_sgn, pred_digits_logits, pred_ops)

    metrics = _run_eval(texts, structures, examples, model)

    # MSE should be strictly positive due to incorrect sign
    assert metrics["final_mse"] > 0.0
