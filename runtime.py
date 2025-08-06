"""Shared runtime utilities and constants.

This module centralises a few small pieces of logic that used to be duplicated
across several entry-point scripts:

* Setting a default `WANDB_DIR` so that local W&B runs are written to /tmp.
* Detecting CUDA / PyTorch capabilities once (CUDA_AVAILABLE).

* Providing a common `CHECKPOINT_DIR` that automatically resolves to the
  correct location when running on RunPod.

Importing this module has **side-effects** (setting the WANDB_DIR env var),
so do it near the top of any script that needs those variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Environment tweaks
# --------------------------------------------------------------------------- #
# Redirect Weights & Biases logs to ephemeral storage by default. This stops the
# persistent project volume from filling up with potentially huge run
# artefacts.
os.environ.setdefault("WANDB_DIR", "/tmp/wandb")

# --------------------------------------------------------------------------- #
# Capability flags
# --------------------------------------------------------------------------- #

CUDA_AVAILABLE: bool = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CHECKPOINT_DIR: Path = (
    Path("/runpod-volume/checkpoints")
    if Path("/runpod-volume").exists()
    else Path("checkpoints")
)
