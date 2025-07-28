"""
Standalone DAG predictor model with shallow attention architecture.

This module contains the PredictorOnlyModel that performs shallow attention
over token embeddings and uses a DAG predictor for structure prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base_model import BaseGPTModel
from .dag_model import OP_NAMES, DAGPlanPredictor


@dataclass
class PredictorOnlyConfig:
    """Configuration for predictor only model."""

    vocab_size: int = 50304  # GPT-2 vocab size
    n_layer: int = 1  # Number of transformer layers
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.0
    bias: bool = False
    dag_depth: int = 4
    block_size: int = 512

    # Digit configuration for initial value prediction
    max_digits: int = 4  # Integer digits
    max_decimal_places: int = 6  # Fractional digits
    base: int = 10  # Number base for digit prediction (10=decimal, 16=hex, etc.)

    # Allowed operation names for DAG execution/prediction. Defaults to the full set.
    op_names: list[str] = field(default_factory=lambda: OP_NAMES.copy())

    # Uncertainty weighting configuration
    train_uncertainty_params: bool = (
        True  # If False, uncertainty params stay at initial values
    )


class PredictorOnlyModel(BaseGPTModel):
    """
    Standalone model that performs attention over token embeddings
    and uses a DAG predictor for structure prediction.

    Architecture:
    Token IDs -> Embeddings -> Position Embeddings -> N Attention Layers -> DAG Predictor
    """

    def __init__(self, config: PredictorOnlyConfig):
        # Validate configuration
        if config.n_layer < 1:
            raise ValueError(f"n_layer must be at least 1, got {config.n_layer}")

        # Call base class constructor which handles transformer creation and weight init
        super().__init__(config)

        # DAG predictor
        self.dag_predictor = DAGPlanPredictor(config)

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass through transformer backbone to DAG predictor.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            DAG predictor outputs (signs, log magnitudes, operation probs)
        """
        # Use inherited transformer backbone
        hidden = self.forward_hidden(input_ids)

        # Apply DAG predictor
        return self.dag_predictor(hidden)
