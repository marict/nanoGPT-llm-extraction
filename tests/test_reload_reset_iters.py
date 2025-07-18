import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

import checkpoint_manager as _cm


def test_reload_reset_iters(tmp_path, monkeypatch):
    """Verify that the `reload_reset_iters` flag controls whether the iteration
    counter from a checkpoint is respected or reset to zero when reloading.
    """

    # ---------------------------------------------------------------------
    # Set up an isolated checkpoint directory and patch the module constant
    # ---------------------------------------------------------------------
    ckpt_dir: Path = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Monkey-patch the global CHECKPOINT_DIR used by CheckpointManager so it
    # looks inside our temporary folder.
    monkeypatch.setattr(_cm, "CHECKPOINT_DIR", str(ckpt_dir))
    # Reload the module so that any module-level properties that captured the
    # old constant (unlikely, but safe) see the new path.
    importlib.reload(_cm)

    cm = _cm.CheckpointManager("regular")

    # ---------------------------------------------------------------------
    # Create a dummy checkpoint file with a non-zero iteration number
    # ---------------------------------------------------------------------
    cfg_name = "pytestcfg"
    initial_iter = 7
    ckpt_path = ckpt_dir / f"ckpt_{cfg_name}_{initial_iter}.pt"
    checkpoint_data = {
        "model_args": {},
        "model": {},
        "iter_num": initial_iter,
        "best_val_loss": 1.23,
    }
    torch.save(checkpoint_data, ckpt_path)

    expected_keys = ["model_args", "model", "iter_num", "best_val_loss"]

    # ------------------------------------------------------------------
    # 1) Standard resume: iteration number should be preserved
    # ------------------------------------------------------------------
    cfg_no_reset = SimpleNamespace(
        name=cfg_name,
        init_from="resume",
        reload_reset_iters=False,
    )

    _, iter_num, _ = cm.handle_checkpoint_loading(
        cfg_no_reset, device="cpu", expected_keys=expected_keys
    )
    assert (
        iter_num == initial_iter
    ), "iter_num should match the value stored in the checkpoint when reload_reset_iters is False"

    # ------------------------------------------------------------------
    # 2) reload_reset_iters=True: iteration should be reset to zero
    # ------------------------------------------------------------------
    cfg_reset = SimpleNamespace(
        name=cfg_name,
        init_from="resume",
        reload_reset_iters=True,
    )

    _, iter_num_reset, _ = cm.handle_checkpoint_loading(
        cfg_reset, device="cpu", expected_keys=expected_keys
    )
    assert (
        iter_num_reset == 0
    ), "iter_num should be reset to 0 when reload_reset_iters is True"
