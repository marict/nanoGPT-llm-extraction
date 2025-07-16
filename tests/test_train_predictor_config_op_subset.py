from pathlib import Path

import pytest
import torch

from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import OP_NAMES
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel
from predictor_config import DAGTrainConfig
from training_utils import load_config_file, update_config

# Path to the RunPod-optimised predictor config
CONFIG_PATH = Path("config/train_predictor_config.py")


def _make_subset_predictor(op_subset: list[str], depth: int) -> PredictorOnlyModel:
    """Build a tiny stand-alone predictor restricted to *op_subset* operations."""

    cfg = PredictorOnlyConfig(
        vocab_size=50,  # tiny vocab for speed
        n_embd=32,
        n_head=2,
        dropout=0.0,
        bias=False,
        dag_depth=depth,
        sequence_length=16,
        op_names=op_subset,
    )
    return PredictorOnlyModel(cfg)


def test_train_predictor_config_op_subset():
    """End-to-end check: config → model → dagset all share the same op subset."""

    # ------------------------------------------------------------------
    # 1. Load Python config file and merge into dataclass
    # ------------------------------------------------------------------
    cfg_dict = load_config_file(str(CONFIG_PATH))
    cfg = DAGTrainConfig()
    update_config(cfg, cfg_dict)

    assert hasattr(cfg, "op_names"), "Config missing 'op_names' field"
    op_subset = cfg.op_names

    # Sanity: ensure subset is valid and non-empty
    assert op_subset, "Config 'op_names' should not be empty"
    assert all(op in OP_NAMES for op in op_subset), "Config contains invalid op name(s)"

    # ------------------------------------------------------------------
    # 2. Build tiny predictor restricted to this subset
    # ------------------------------------------------------------------
    model = _make_subset_predictor(op_subset, depth=cfg.dag_depth)
    model.eval()

    # Verify model knows about the same subset
    assert (
        model.dag_predictor.op_names == op_subset
    ), "Model's allowed ops diverge from config"

    # ------------------------------------------------------------------
    # 3. Create dataloader using *model*'s allowed operations
    # ------------------------------------------------------------------
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=4,
        val_batch_size=4,
        max_depth=cfg.dag_depth,
        seed=cfg.seed,
        english_conversion_probability=cfg.english_conversion_probability,
        integer_no_decimal_probability=cfg.integer_no_decimal_probability,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        allowed_operations=model.dag_predictor.op_names,
    )

    # Grab a single batch
    _texts, structures, _ = next(train_loader)
    op_one_hot = structures["operation_probs"]  # (B, depth, len(OP_NAMES))
    picked_indices = op_one_hot.argmax(dim=-1).unique().tolist()
    allowed_global_indices = [OP_NAMES.index(op) for op in op_subset]

    assert set(picked_indices).issubset(
        set(allowed_global_indices)
    ), "Dataset produced operations outside the model/config subset"

    # ------------------------------------------------------------------
    # 4. Forward pass through predictor and ensure logits cover *only* subset
    # ------------------------------------------------------------------
    input_ids = torch.randint(
        0, model.config.vocab_size, (2, model.config.sequence_length)
    )
    with torch.no_grad():
        _sgn, _log, op_probs = model(input_ids)  # (B, T, depth, |subset|)

    # Predictor should now produce probabilities for the *full* OP_NAMES set
    # with zeros for disallowed operations.
    assert op_probs.shape[-1] == len(OP_NAMES)

    # All probability mass must reside within the allowed subset columns.
    disallowed_indices = {
        idx for idx in range(len(OP_NAMES)) if idx not in allowed_global_indices
    }
    probs_sum_disallowed = op_probs[..., list(disallowed_indices)].sum()
    assert (
        probs_sum_disallowed.item() < 1e-5
    ), "Predictor assigned mass to disallowed operations"
