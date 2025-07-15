"""
Standalone DAG predictor model with shallow attention architecture.

This module contains the PredictorOnlyModel that performs shallow attention
over token embeddings and uses a DAG predictor for structure prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .dag_model import OP_NAMES, Block, DAGPlanPredictor, LayerNorm


@dataclass
class PredictorOnlyConfig:
    """Configuration for predictor only model."""

    vocab_size: int = 50304  # GPT-2 vocab size
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.0
    bias: bool = False
    dag_depth: int = 4
    sequence_length: int = 512
    softmax_temperature: float = 20.0

    # Allowed operation names for DAG execution/prediction. Defaults to the full set.
    op_names: list[str] = field(default_factory=lambda: OP_NAMES.copy())


class PredictorOnlyModel(nn.Module):
    """
    Standalone model that performs shallow attention over token embeddings
    and uses a DAG predictor for structure prediction.

    Architecture:
    Token IDs -> Embeddings -> Position Embeddings -> Single Attention Layer -> DAG Predictor
    """

    def __init__(self, config: PredictorOnlyConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Single attention block for shallow attention
        self.attention_block = Block(config)

        # Layer norm for stability
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # DAG predictor
        self.dag_predictor = DAGPlanPredictor(config, config.softmax_temperature)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass through shallow attention to DAG predictor.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            DAG predictor outputs (signs, log magnitudes, operation probs)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.wte(input_ids)  # (B, T, n_embd)

        # Position embeddings
        pos = torch.arange(T, device=device)
        pos_emb = self.wpe(pos)  # (T, n_embd)

        # Combine embeddings
        hidden = self.drop(token_emb + pos_emb)  # (B, T, n_embd)

        # Single attention layer (shallow attention)
        hidden = self.attention_block(hidden)  # (B, T, n_embd)

        # Final layer norm
        hidden = self.ln_f(hidden)  # (B, T, n_embd)

        # DAG predictor
        return self.dag_predictor(hidden)

    def get_num_params(self, non_embedding=True):
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
