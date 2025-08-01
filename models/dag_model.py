"""
Full definition of a GPT Language Model with optional DAG augmentation.
When dag_depth=0, behaves as a standard GPT model.
When dag_depth>0, uses differentiable ALU DAG for enhanced reasoning.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from rff.layers import PositionalEncoding
from torch.nn import functional as F

from .base_model import BaseGPTModel

# Log-space arithmetic utilities
LOG_LIM = (
    100.0  # Bound on ln-magnitudes (increased to handle extreme large expressions)
)
MIN_CLAMP = 1e-6

# Debug utility
ENABLE_DEBUG_NAN_CHECKS = os.getenv("DAGGPT_DEBUG_NANS", "0") == "1"
if ENABLE_DEBUG_NAN_CHECKS:
    print("dag_model: DEBUG_NAN_CHECKS is enabled")


class DAGExecutor(nn.Module):
    """New tensor-based DAG execution engine with 50/50 node split architecture."""

    def __init__(self, dag_depth: int):
        super().__init__()
        self.dag_depth = dag_depth
        self.total_nodes = dag_depth * 2
        self.initial_slots = dag_depth
        self.intermediate_slots = dag_depth

    def execute_with_plan(self, V_mag, V_sign, O, G):
        """Execute DAG with given tensors."""
        return self.forward(V_mag, V_sign, O, G)

    def forward(self, V_mag, V_sign, O, G):
        """
            Execute DAG operations using tensor representation with 50/50 node split.

        Args:
                V_mag: (1, 1, total_nodes) - magnitudes of all nodes (initial + intermediate)
                V_sign: (1, 1, total_nodes) - signs of all nodes
                O: (1, 1, dag_depth, total_nodes) - operand selection matrix
                G: (1, 1, dag_depth) - domain selector (0=log, 1=linear)

        Returns:
                torch.Tensor: final result (scalar)
        """
        _, _, total_nodes = V_mag.shape

        # Store original dtype for final result
        original_dtype = V_mag.dtype

        # Cast to float64 for precise computation (except on MPS)
        compute_dtype = torch.float64 if V_mag.device.type != "mps" else torch.float32

        # Work with a copy to avoid modifying the input and cast to high precision
        working_V_mag = V_mag.clone().to(compute_dtype)
        working_V_sign = V_sign.clone().to(compute_dtype)

        # Execute each intermediate computation step
        for step in range(self.intermediate_slots):
            # Get operand selector and domain gate for this step
            O_step = O[:, :, step, :].to(compute_dtype)  # (B, T, total_nodes)
            G_step = G[:, :, step].unsqueeze(-1).to(compute_dtype)  # (B, T, 1)

            # Apply triangular mask to prevent using future intermediate results
            valid_positions = (
                self.initial_slots + step
            )  # How many positions are available

            # Create causal mask
            causal_mask = torch.zeros_like(O_step)  # (B, T, total_nodes)
            causal_mask[:, :, :valid_positions] = 1.0

            # Apply mask to operand selector
            O_step = O_step * causal_mask

            # Domain-mixed computation (keeping original dtype for gradients)
            signed_values = working_V_sign * working_V_mag
            log_mag = torch.log(torch.clamp(working_V_mag, min=1e-12))
            mixed = log_mag * (1 - G_step) + signed_values * G_step
            R_mag = torch.sum(O_step * mixed, dim=-1, keepdim=True)

            # Linear domain sign
            linear_sign = torch.tanh(R_mag / 0.0001)

            # Log domain sign: product of selected signs.
            # We multiply by 2 and add 1 such that
            # pos: 2*1 + 1 = 3
            # neg: 2*(-1) + 1 = -1
            # zero: 2*0 + 1 = 1 (does not change sign)
            sign_weights = (working_V_sign * torch.abs(O_step)) * 2 + 1
            sign_product = torch.prod(sign_weights, dim=-1, keepdim=True)
            log_sign = torch.tanh(sign_product / 0.0001)

            V_sign_new = G_step * linear_sign + (1 - G_step) * log_sign

            # For magnitude: linear domain uses abs(R_mag), log domain uses exp(R_mag)
            linear_mag = torch.abs(R_mag)
            log_mag_result = torch.exp(torch.clamp(R_mag, max=LOG_LIM))
            V_mag_new = G_step * linear_mag + (1 - G_step) * log_mag_result

            # Write result to the predetermined intermediate slot
            intermediate_idx = self.initial_slots + step
            # Use scatter to avoid in-place operations that break gradients
            indices = torch.tensor(
                [intermediate_idx], device=working_V_mag.device
            ).expand(working_V_mag.shape[:2])
            working_V_mag = working_V_mag.scatter(-1, indices.unsqueeze(-1), V_mag_new)
            working_V_sign = working_V_sign.scatter(
                -1, indices.unsqueeze(-1), V_sign_new
            )

        # Always use the last intermediate slot as the final result
        final_idx = total_nodes - 1
        final_mag = working_V_mag[:, :, final_idx]  # (B, T)
        final_sign = working_V_sign[:, :, final_idx]  # (B, T)
        final_value = final_sign * final_mag  # (B, T)

        # Cast back to original dtype for gradient compatibility
        return final_value.to(original_dtype)


class DAGPlanPredictor(nn.Module):
    """Predictor for tensor-based DAG execution with 50/50 node split architecture."""

    def __init__(self, config):
        super().__init__()
        self.dag_depth = config.dag_depth
        self.total_nodes = (
            config.dag_depth * 2
        )  # 50/50 split: initial + intermediate nodes
        self.n_embd = config.n_embd

        # Predict tensor components with correct 50/50 architecture
        # V_mag: magnitudes for all nodes (initial + intermediate)
        self.V_mag_predictor = nn.Linear(config.n_embd, self.total_nodes)

        # V_sign: signs for all nodes
        self.V_sign_predictor = nn.Linear(config.n_embd, self.total_nodes)

        # O: operand selection matrix (dag_depth operations × total_nodes)
        self.O_predictor = nn.Linear(config.n_embd, config.dag_depth * self.total_nodes)

        # G: domain selector for operations
        self.G_predictor = nn.Linear(config.n_embd, config.dag_depth)

        # Initialize weights
        for module in [
            self.V_mag_predictor,
            self.V_sign_predictor,
            self.O_predictor,
            self.G_predictor,
        ]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(module.bias)

    def forward(self, hidden_state):
        """
            Predict DAG tensors from causal hidden states (T separate DAGs).

            Args:
                hidden_state: (B, T, n_embd) - causal hidden states from transformer

        Returns:
                V_mag: (B, T, total_nodes) - magnitudes for all nodes
                V_sign: (B, T, total_nodes) - signs for all nodes
                O: (B, T, dag_depth, total_nodes) - operand selection matrix
                G: (B, T, dag_depth) - domain selector
        """
        B, T = hidden_state.shape[:2]

        # Predict each component with correct 50/50 architecture
        V_mag = torch.abs(self.V_mag_predictor(hidden_state))  # (B, T, total_nodes)
        V_sign = torch.tanh(self.V_sign_predictor(hidden_state))  # (B, T, total_nodes)

        # Predict operand matrix and reshape
        O_flat = self.O_predictor(hidden_state)  # (B, T, dag_depth * total_nodes)
        O = O_flat.view(
            B, T, self.dag_depth, self.total_nodes
        )  # (B, T, dag_depth, total_nodes)

        # Domain selector: 0=log, 1=linear
        G = torch.sigmoid(self.G_predictor(hidden_state))  # (B, T, dag_depth)

        return V_mag, V_sign, O, G


class ScalarToEmbed(nn.Module):
    """Map sign & log magnitude to embedding using Fourier bases."""

    def __init__(self, hidden: int, sigma=1.0, feat_dim: int = 32):
        super().__init__()
        self.ff = PositionalEncoding(sigma, feat_dim)
        self.proj = nn.Linear(2 * feat_dim + 2, hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Convert scalar features to embeddings."""
        # feat is expected to be (B, 2) with [sign, log_magnitude]
        sign = feat[..., 0:1]  # (B, 1)
        log_mag = feat[..., 1:2]  # (B, 1)

        # Apply Fourier features to log magnitude
        ff_features = self.ff(log_mag)  # (B, 2*feat_dim)

        # Concatenate sign, log_mag, and Fourier features
        combined = torch.cat(
            [sign, log_mag, ff_features], dim=-1
        )  # (B, 2 + 2*feat_dim)

        # Project to hidden dimension
        return self.proj(combined)  # (B, hidden)


@dataclass
class GPTConfig:
    # Model architecture
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )

    # DAG-specific parameters
    dag_depth: int = 6


class GPT(BaseGPTModel):
    """GPT model with DAG computation capability."""

    def __init__(self, config):
        super().__init__(config)

        if config.dag_depth > 0:
            # Use DAG system
            self.dag_predictor = DAGPlanPredictor(config)
            self.dag_executor = DAGExecutor(config.dag_depth)
        else:
            self.dag_predictor = None
            self.dag_executor = None

    def forward(self, idx, targets=None):
        """
        Forward pass with DAG computation.

        Args:
            idx: (B, T) input token indices
            targets: (B, T) target values for loss computation (optional)

        Returns:
            If targets is None: (B, T) DAG predictions
            If targets is provided: (loss, (B, T) DAG predictions)
        """
        # Get causal hidden states from transformer backbone
        hidden_states = self.forward_hidden(idx)  # (B, T, n_embd)

        if self.dag_predictor is not None:
            # Predict DAG tensors for each token position
            V_mag, V_sign, O, G = self.dag_predictor(hidden_states)  # (B, T, ...)

            # Execute DAG for each token position
            B, T = hidden_states.shape[:2]
            dag_outputs = []

            for t in range(T):
                # Execute DAG for token t using only causal information
                V_mag_t = V_mag[:, t : t + 1, :]  # (B, 1, total_nodes)
                V_sign_t = V_sign[:, t : t + 1, :]  # (B, 1, total_nodes)
                O_t = O[:, t : t + 1, :, :]  # (B, 1, dag_depth, total_nodes)
                G_t = G[:, t : t + 1, :]  # (B, 1, dag_depth)

                result_t = self.dag_executor.execute_with_plan(
                    V_mag_t, V_sign_t, O_t, G_t
                )  # (B, 1)
                dag_outputs.append(result_t)

            # Concatenate results
            predictions = torch.cat(dag_outputs, dim=1)  # (B, T)

            if targets is not None:
                # Compute loss
                loss = F.mse_loss(predictions, targets)
                return loss, predictions
            else:
                return predictions
        else:
            # No DAG - this shouldn't happen in practice
            raise ValueError("DAG depth is 0, cannot perform DAG computation")
