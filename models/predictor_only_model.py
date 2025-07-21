"""
Standalone DAG predictor model with shallow attention architecture.

This module contains the PredictorOnlyModel that performs shallow attention
over token embeddings and uses a DAG predictor for structure prediction.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .dag_model import OP_NAMES, Block, DAGPlanPredictor, LayerNorm


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
    sequence_length: int = 512

    # Digit configuration for initial value prediction
    max_digits: int = 4  # Integer digits
    max_decimal_places: int = 6  # Fractional digits

    # Allowed operation names for DAG execution/prediction. Defaults to the full set.
    op_names: list[str] = field(default_factory=lambda: OP_NAMES.copy())


class PredictorOnlyModel(nn.Module):
    """
    Standalone model that performs attention over token embeddings
    and uses a DAG predictor for structure prediction.

    Architecture:
    Token IDs -> Embeddings -> Position Embeddings -> N Attention Layers -> DAG Predictor
    """

    def __init__(self, config: PredictorOnlyConfig):
        super().__init__()
        self.config = config

        # Validate configuration
        if config.n_layer < 1:
            raise ValueError(f"n_layer must be at least 1, got {config.n_layer}")

        # Transformer backbone (using same naming as GPT model for easy checkpoint loading)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # DAG predictor
        self.dag_predictor = DAGPlanPredictor(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            # Use Glorot/Xavier for operations_projection weights and preserve its bias
            if getattr(module, "is_operations_projection", False):
                torch.nn.init.xavier_uniform_(module.weight)
                # Bias was pre-initialised (-4.0) in DAGPlanPredictor; keep as-is.
            elif getattr(module, "is_initial_values_predictor", False):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Bias was pre-initialised with digit biases in DAGPlanPredictor; keep as-is.
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, idx):
        """
        Forward pass through transformer backbone only, returning hidden states.

        Args:
            idx: (B, T) token IDs

        Returns:
            Hidden states after transformer backbone (B, T, n_embd)
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, sequence length is only {self.config.sequence_length}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass through configurable attention layers to DAG predictor.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            DAG predictor outputs (signs, log magnitudes, operation probs)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Validate sequence length
        assert (
            T <= self.config.sequence_length
        ), f"Cannot forward sequence of length {T}, sequence length is only {self.config.sequence_length}"

        # Token embeddings
        token_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)

        # Position embeddings
        pos = torch.arange(T, device=device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # Combine embeddings
        hidden = self.transformer.drop(token_emb + pos_emb)  # (B, T, n_embd)

        # Apply all attention blocks
        for block in self.transformer.h:
            hidden = block(hidden)  # (B, T, n_embd)

        # Final layer norm
        hidden = self.transformer.ln_f(hidden)  # (B, T, n_embd)

        # DAG predictor
        return self.dag_predictor(hidden)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def crop_sequence_length(self, sequence_length):
        """Adjust model to use a smaller sequence length if needed."""
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:sequence_length]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[
                    :, :, :sequence_length, :sequence_length
                ]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with weight decay for 2D parameters only."""
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optimizer groups: 2D params get weight decay, others don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Use fused AdamW if available on CUDA
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # Estimate FLOPs per iteration (see PaLM paper Appendix B)
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.sequence_length,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express as ratio of A100 peak FLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS
        mfu = flops_achieved / flops_promised
        return mfu
