"""Shared runtime utilities and constants.

This module centralises a few small pieces of logic that used to be duplicated
across several entry-point scripts:

* Setting a default `WANDB_DIR` so that local W&B runs are written to /tmp.
* Detecting CUDA / PyTorch capabilities once (TORCH_2_2_1, CUDA_AVAILABLE).
* Detecting whether the `safetensors` package is available.
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
TORCH_2_2_1: bool = torch.__version__ >= "2.2.1"
CUDA_AVAILABLE: bool = torch.cuda.is_available()

# Optional safetensors for pure-tensor checkpoints
try:
    import safetensors.torch as _st  # type: ignore

    _HAVE_ST = True
except ModuleNotFoundError:  # pragma: no cover – safe fallback
    _HAVE_ST = False

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CHECKPOINT_DIR: Path = (
    Path("/runpod-volume/checkpoints")
    if Path("/runpod-volume").exists()
    else Path("checkpoints")
)
