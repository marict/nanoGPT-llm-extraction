from pathlib import Path

import pytest

from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from training_utils import load_config_file, update_config

DEFAULT_CFG_PATH = Path("config/train_predictor_default.py")


def test_train_predictor_default_config_op_subset():
    """The default train_predictor config should restrict DAG ops to its op_names subset."""

    # 1. Load Python config file and merge into DAGTrainConfig
    cfg_dict = load_config_file(str(DEFAULT_CFG_PATH))
    cfg = DAGTrainConfig()
    update_config(cfg, cfg_dict)

    assert hasattr(
        cfg, "op_names"
    ), "DAGTrainConfig is missing 'op_names' after update."

    # Sanity: config file sets a subset (add, subtract, identity)
    expected_subset = cfg_dict.get("op_names")
    assert expected_subset is not None, "Default config should define op_names"
    assert cfg.op_names == expected_subset

    # 2. Create a tiny dataloader using the same helper as train_predictor
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=4,
        val_batch_size=4,
        max_depth=cfg.dag_depth,
        seed=cfg.seed,
        english_conversion_rate=cfg.english_conversion_rate,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        allowed_operations=cfg.op_names,
    )

    # 3. Grab a single batch and verify all operations are within subset
    texts, structures, _ = next(train_loader)
    op_one_hot = structures["operation_probs"]  # shape (B, depth, len(OP_NAMES))
    picked_indices = op_one_hot.argmax(dim=-1).unique().tolist()
    allowed_indices = [OP_NAMES.index(op) for op in cfg.op_names]

    assert set(picked_indices).issubset(
        set(allowed_indices)
    ), "DataLoader produced operations outside the expected subset."
