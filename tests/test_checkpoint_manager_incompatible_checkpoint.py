import types

import pytest

from checkpoint_manager import CheckpointLoadError, CheckpointManager
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def _make_cfg(**kwargs):
    """Utility to create a SimpleNamespace config with sensible defaults."""
    defaults = {
        "n_embd": 64,
        "n_head": 4,
        "dropout": 0.0,
        "bias": False,
        "dag_depth": 4,
        "max_digits": 2,
        "max_decimal_places": 2,
        "block_size": 32,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_initialize_dag_model_ignores_incompatible_checkpoint():
    """When critical config values differ, the checkpoint should be ignored."""
    # Original (saved) checkpoint with dag_depth = 4
    saved_cfg_dict = {
        "vocab_size": 50304,
        "n_embd": 64,
        "n_head": 4,
        "dropout": 0.0,
        "bias": False,
        "dag_depth": 4,
        "max_digits": 2,
        "max_decimal_places": 2,
        "block_size": 32,
    }
    saved_model_cfg = PredictorOnlyConfig(**saved_cfg_dict)
    saved_model = PredictorOnlyModel(saved_model_cfg)
    checkpoint = {
        "model": saved_model.state_dict(),
        "model_config": saved_cfg_dict,
    }

    # Current run config with **different** dag_depth (6 instead of 4)
    current_cfg = _make_cfg(dag_depth=6)

    manager = CheckpointManager(checkpoint_type="dag")

    # Expect a CheckpointLoadError due to incompatible dag_depth
    with pytest.raises(CheckpointLoadError):
        _ = manager.initialize_dag_model(current_cfg, checkpoint)
