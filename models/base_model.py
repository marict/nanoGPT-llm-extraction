"""
Shared base class for GPT-style models with common transformer backbone functionality.

This module provides a BaseGPTModel that contains shared logic for transformer
architectures, eliminating code duplication between different model variants.
"""

import inspect
import math

import torch
import torch.nn as nn

from .components import Block, LayerNorm


class BaseGPTModel(nn.Module):
    """
    Base class for GPT-style models with shared transformer backbone.

    Provides common functionality including:
    - Transformer backbone creation
    - Weight initialization
    - Parameter counting
    - Optimizer configuration
    - MFU estimation
    - Forward hidden computation
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create transformer backbone using common naming convention
        self.transformer = self._create_transformer_backbone(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self._get_n_layer())
                )

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _create_transformer_backbone(self, config):
        """Create the transformer backbone with standard components."""
        return nn.ModuleDict(
            dict(
                wte=nn.Embedding(self._get_vocab_size(), self._get_n_embd()),
                wpe=nn.Embedding(self._get_max_sequence_length(), self._get_n_embd()),
                drop=nn.Dropout(self._get_dropout()),
                h=nn.ModuleList([Block(config) for _ in range(self._get_n_layer())]),
                ln_f=LayerNorm(self._get_n_embd(), bias=self._get_bias()),
            )
        )

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization.

        Note: DAGPlanPredictor handles its own specialized initialization internally.
        """
        # Skip initialization for DAGPlanPredictor modules to preserve their custom initialization
        if self._should_skip_dag_predictor_init(module):
            return

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _should_skip_dag_predictor_init(self, module):
        """Check if module should skip initialization (for DAG predictor components)."""
        # Check if this model has DAG predictor components and if this module is part of them
        dag_predictor_attrs = ["dag_predictor", "plan_predictor"]

        for attr_name in dag_predictor_attrs:
            if hasattr(self, attr_name):
                dag_predictor = getattr(self, attr_name)
                # Handle case where dag_predictor property returns None for dag_depth=0
                if dag_predictor is not None:
                    for name, dag_module in dag_predictor.named_modules():
                        if module is dag_module:
                            return True

        # Special case for dag.plan_predictor in GPT model
        if hasattr(self, "dag") and hasattr(self.dag, "plan_predictor"):
            for name, dag_module in self.dag.plan_predictor.named_modules():
                if module is dag_module:
                    return True

        return False

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
        max_length = self._get_max_sequence_length()
        assert (
            t <= max_length
        ), f"Cannot forward sequence of length {t}, max sequence length is only {max_length}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

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
        L, H, Q, T = (
            self._get_n_layer(),
            self._get_n_head(),
            self._get_n_embd() // self._get_n_head(),
            self._get_max_sequence_length(),
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express as ratio of A100 peak FLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def crop_sequence_length(self, new_length):
        """Adjust model to use a smaller sequence length if needed."""
        max_length = self._get_max_sequence_length()
        assert new_length <= max_length

        # Update config
        self._set_max_sequence_length(new_length)

        # Crop position embeddings
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:new_length]
        )

        # Crop attention bias matrices if they exist
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :new_length, :new_length]

    # Default config access implementations (identical across all models)
    def _get_vocab_size(self):
        """Return vocabulary size from config."""
        return self.config.vocab_size

    def _get_n_embd(self):
        """Return embedding dimension from config."""
        return self.config.n_embd

    def _get_n_layer(self):
        """Return number of layers from config."""
        return self.config.n_layer

    def _get_n_head(self):
        """Return number of attention heads from config."""
        return self.config.n_head

    def _get_dropout(self):
        """Return dropout rate from config."""
        return self.config.dropout

    def _get_bias(self):
        """Return bias setting from config."""
        return self.config.bias

    def _get_max_sequence_length(self):
        """Return maximum sequence length from config."""
        return self.config.block_size

    def _set_max_sequence_length(self, new_length):
        """Update maximum sequence length in config."""
        self.config.block_size = new_length
