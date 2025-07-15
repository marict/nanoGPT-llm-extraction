from pathlib import Path

from checkpoint_manager import CheckpointManager
from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from training_utils import load_config_file, update_config

CONFIG_PATH = Path("config/train_predictor_config.py")


def test_checkpoint_manager_propagates_op_subset():
    """CheckpointManager.initialize_dag_model should propagate cfg.op_names to model config."""

    cfg_dict = load_config_file(str(CONFIG_PATH))
    cfg = DAGTrainConfig()
    update_config(cfg, cfg_dict)

    # Basic sanity
    subset = cfg.op_names
    assert subset, "Config 'op_names' must not be empty"
    assert all(op in OP_NAMES for op in subset)

    ckpt_mgr = CheckpointManager("dag")

    # Initialise model (no checkpoint) and verify predictor's op_names
    model, model_config = ckpt_mgr.initialize_dag_model(
        cfg, checkpoint=None, device="cpu"
    )

    # Access underlying predictor module (handles DDP etc.)
    if hasattr(model, "dag_predictor"):
        op_names_model = model.dag_predictor.op_names
    elif hasattr(model, "dag") and hasattr(model.dag, "plan_predictor"):
        op_names_model = model.dag.plan_predictor.op_names
    else:
        raise RuntimeError("Unexpected model architecture in test")

    assert op_names_model == subset, (
        "initialize_dag_model did not pass operation subset to the predictor; "
        "model can predict disallowed operations"
    )
