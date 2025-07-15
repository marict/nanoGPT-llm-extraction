from __future__ import annotations

"""Configuration dataclass for DAG predictor pre-training.
Extracted from *train_predictor.py* to avoid code duplication and shrink the main
training script by over a hundred lines.
"""

from dataclasses import dataclass, field

# Import operation names for default value
from models.dag_model import OP_NAMES
from training_utils import BaseConfig


@dataclass
class DAGTrainConfig(BaseConfig):
    """Container for DAG predictor training hyperparameters."""

    # Meta
    name: str = "dag_pretrain"

    # Learning rate schedule
    use_cyclical_lr: bool = False
    cyclical_lr_period: int = 1000
    cyclical_lr_amplitude: float = 0.1

    # DAG dataset parameters
    max_dag_depth: int = 8
    max_digits: int = 4
    max_decimal_places: int | None = None
    train_examples_per_batch: int = 1000
    val_examples_per_batch: int = 100

    # English conversion
    english_conversion_rate: float = 0.3

    # Batching / sequence
    gradient_accumulation_steps: int = 4
    batch_size: int = 32
    sequence_length: int = 512

    # Model architecture
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    dag_depth: int = 4

    # Optimisation
    learning_rate: float = 3e-4
    max_iters: int = 10000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 500
    lr_decay_iters: int = 10000
    min_lr: float = 3e-5

    # Loss weights
    sign_loss_weight: float = 1.0
    digit_loss_weight: float = 1.0
    op_loss_weight: float = 1.0

    # Random seeds
    seed: int = 42

    # Allowed operation names for DAG execution/prediction. This lets us train
    # predictors restricted to a subset of arithmetic ops (e.g. only add, subtract, identity).
    op_names: list[str] = field(default_factory=lambda: OP_NAMES.copy())

    # Backbone options
    full_backbone: bool = False
    n_layer: int = 12
