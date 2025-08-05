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

from dataclasses import dataclass

import torch
import torch.nn as nn
from rff.layers import PositionalEncoding

from data.dagset.streaming import tensor_to_expression

from .base_model import BaseGPTModel

# Log-space arithmetic utilities
LOG_LIM = (
    100.0  # Bound on ln-magnitudes (increased to handle extreme large expressions)
)

# Magnitude clamping constants
MAG_MIN = 1e-12  # Minimum magnitude for log operations and intermediate results
MAG_MAX = 1e28  # Maximum magnitude ceiling to catch overflow bugs
SIGN_MIN = -1.0  # Minimum sign value
SIGN_MAX = 1.0  # Maximum sign value


class DAGExecutor(nn.Module):
    """New tensor-based DAG execution engine with (dag_depth + 1) + dag_depth node architecture."""

    def __init__(
        self,
        dag_depth: int,
        max_digits: int = 4,
        max_decimal_places: int = 4,
        base: int = 10,
    ):
        super().__init__()
        self.dag_depth = dag_depth
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        self.base = base
        self.num_initial_nodes = dag_depth + 1
        self.num_intermediate_nodes = dag_depth
        self.total_nodes = self.num_initial_nodes + self.num_intermediate_nodes

    @staticmethod
    def digits_to_vmag(
        digit_logits: torch.Tensor,
        max_digits: int,
        base: int = 10,
        temperature: float = 0.01,
        apply_ste: bool = False,
    ) -> torch.Tensor:
        """Convert digit logits to magnitude values using vectorized operations."""
        _, _, _, D, _ = digit_logits.shape
        device, dtype = digit_logits.device, digit_logits.dtype

        # Convert logits to expected digit values
        digit_probs = torch.softmax(digit_logits / temperature, dim=-1)

        if apply_ste:
            # Apply STE for digits: hard one-hot in forward, soft gradients in backward
            # Use a small temperature to make argmax more stable
            hard_probs = torch.softmax(digit_logits / 0.1, dim=-1)
            digit_one_hot = torch.nn.functional.one_hot(
                hard_probs.argmax(dim=-1), num_classes=base
            ).float()
            # The STE magic: forward uses digit_one_hot, backward uses digit_probs
            digit_probs = digit_one_hot + (digit_probs - digit_probs.detach())

        digit_values = torch.arange(base, dtype=dtype, device=device)
        expected_digits = (digit_probs * digit_values.view(1, 1, 1, 1, -1)).sum(dim=-1)

        # Create positional weights for base conversion (vectorized)
        int_powers = torch.tensor(
            [base ** (max_digits - 1 - d) for d in range(max_digits)],
            dtype=dtype,
            device=device,
        )
        frac_powers = torch.tensor(
            [base ** (max_digits - 1 - d) for d in range(max_digits, D)],
            dtype=dtype,
            device=device,
        )

        # Vectorized base conversion
        int_part = (expected_digits[..., :max_digits] * int_powers).sum(dim=-1)
        frac_part = (expected_digits[..., max_digits:] * frac_powers).sum(dim=-1)

        # Combine and clamp to prevent overflow/underflow
        V_mag = torch.clamp(int_part + frac_part, min=MAG_MIN, max=MAG_MAX)

        return V_mag

    @staticmethod
    def ste_round(x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator for rounding a tensor to the nearest integer."""
        return x.round().detach() + (x - x.detach())

    def _compute_domain_mixed_result(
        self, working_V_mag, working_V_sign, O_step, G_step
    ):
        """Compute domain-mixed arithmetic result for one step."""
        signed_values = working_V_sign * working_V_mag
        log_mag = torch.log(torch.clamp(working_V_mag, min=MAG_MIN))
        mixed = log_mag * (1 - G_step) + signed_values * G_step
        return torch.sum(O_step * mixed, dim=-1, keepdim=True)

    def _compute_new_sign(self, R_mag, working_V_sign, O_step, G_step):
        """Compute new sign using domain mixing."""
        # Linear domain sign
        linear_sign = torch.tanh(R_mag / 0.0001)

        # Log domain sign (product of selected signs)
        sign_weights = (working_V_sign * torch.abs(O_step)) * 2 + 1
        log_sign = torch.tanh(torch.prod(sign_weights, dim=-1, keepdim=True) / 0.0001)

        return G_step * linear_sign + (1 - G_step) * log_sign

    def _compute_new_magnitude(self, R_mag, G_step):
        """Compute new magnitude using domain mixing."""
        linear_mag = torch.clamp(torch.abs(R_mag), max=MAG_MAX)
        R_mag_clamped = torch.clamp(R_mag, min=-LOG_LIM, max=LOG_LIM)
        log_mag_result = torch.exp(R_mag_clamped)
        return G_step * linear_mag + (1 - G_step) * log_mag_result

    def forward(self, digit_logits, V_sign, O, G, apply_ste: bool = False):
        """Execute DAG operations using tensor representation."""
        B, T = V_sign.shape[:2]

        # Store original dtype for final result
        original_dtype = V_sign.dtype

        # Cast to float64 for precise computation (except on MPS)
        compute_dtype = torch.float64 if V_sign.device.type != "mps" else torch.float32

        # Convert digit predictions to V_mag for initial nodes using shared method
        initial_V_mag = self.digits_to_vmag(
            digit_logits, self.max_digits, self.base
        )  # (B, T, num_initial_nodes)

        # Create full working tensors
        working_V_mag = torch.zeros(
            B, T, self.total_nodes, dtype=compute_dtype, device=V_sign.device
        )
        working_V_sign = V_sign.clone().to(compute_dtype)

        # Copy initial node magnitudes into working tensor (convert dtype if needed)
        working_V_mag[:, :, : self.num_initial_nodes] = initial_V_mag.to(compute_dtype)

        O = O.to(compute_dtype)
        G = G.to(compute_dtype)

        # Execute each intermediate computation step
        for step in range(self.num_intermediate_nodes):
            # Get operand selector and domain gate for this step
            O_step = O[:, :, step, :]
            G_step = G[:, :, step].unsqueeze(-1)

            # Apply causal mask to prevent using future intermediate results
            valid_positions = self.num_initial_nodes + step
            causal_mask = torch.zeros_like(O_step)
            causal_mask[:, :, :valid_positions] = 1.0
            O_step = O_step * causal_mask

            # Compute new values using helper functions
            R_mag = self._compute_domain_mixed_result(
                working_V_mag, working_V_sign, O_step, G_step
            )
            V_sign_new = self._compute_new_sign(R_mag, working_V_sign, O_step, G_step)
            V_mag_new = self._compute_new_magnitude(R_mag, G_step)

            # Clamp intermediate results
            V_mag_new = torch.clamp(V_mag_new, min=MAG_MIN, max=MAG_MAX)
            V_sign_new = torch.clamp(V_sign_new, min=SIGN_MIN, max=SIGN_MAX)

            # Update working tensors (use scatter to avoid in-place operations)
            intermediate_idx = self.num_initial_nodes + step
            indices = torch.tensor(
                [intermediate_idx], device=working_V_mag.device
            ).expand(working_V_mag.shape[:2])
            working_V_mag = working_V_mag.scatter(-1, indices.unsqueeze(-1), V_mag_new)
            working_V_sign = working_V_sign.scatter(
                -1, indices.unsqueeze(-1), V_sign_new
            )

        # Always use the last intermediate slot as the final result
        final_idx = self.total_nodes - 1
        final_mag = working_V_mag[:, :, final_idx]  # (B, T)
        final_sign = working_V_sign[:, :, final_idx]  # (B, T)
        final_value = final_sign * final_mag  # (B, T)

        # Cast back to original dtype
        return final_value.to(original_dtype)


class DAGPlanPredictor(nn.Module):
    """Predictor for tensor-based DAG execution with (dag_depth + 1) + dag_depth node architecture using digit prediction."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dag_depth = config.dag_depth
        self.max_digits = config.max_digits
        self.max_decimal_places = config.max_decimal_places
        self.base = 10  # Default base

        num_initial_nodes = config.dag_depth + 1
        num_intermediate_nodes = config.dag_depth
        self.num_initial_nodes = num_initial_nodes

        self.total_nodes = num_initial_nodes + num_intermediate_nodes
        self.n_embd = config.n_embd

        # Digit dimensions
        self.D = self.max_digits + self.max_decimal_places

        # Predict tensor components with (dag_depth + 1) + dag_depth architecture
        # digit_predictor: digit logits for initial nodes only (magnitudes via digits)
        self.digit_predictor = nn.Linear(
            config.n_embd, self.num_initial_nodes * self.D * self.base
        )

        # V_sign: signs for all nodes
        self.V_sign_predictor = nn.Linear(config.n_embd, self.total_nodes)

        # O: operand selection matrix (dag_depth operations × total_nodes)
        self.O_predictor = nn.Linear(config.n_embd, config.dag_depth * self.total_nodes)

        # G: domain selector for operations
        self.G_predictor = nn.Linear(config.n_embd, config.dag_depth)

        # Initialize weights
        for module in [
            self.digit_predictor,
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
                digit_logits: (B, T, num_initial_nodes, D, base) - digit logits for initial nodes
                V_sign: (B, T, total_nodes) - signs for all nodes
                O: (B, T, dag_depth, total_nodes) - operand selection matrix
                G: (B, T, dag_depth) - domain selector
        """
        B, T = hidden_state.shape[:2]

        # Predict digit logits for initial nodes only
        digit_flat = self.digit_predictor(
            hidden_state
        )  # (B, T, num_initial_nodes * D * base)
        digit_logits = digit_flat.view(
            B, T, self.num_initial_nodes, self.D, self.base
        )  # (B, T, num_initial_nodes, D, base)

        # V_sign: signs for all nodes
        V_sign = torch.tanh(self.V_sign_predictor(hidden_state))  # (B, T, total_nodes)

        # Predict operand matrix and reshape
        # We don't tanh this because we need to have coefficients |x| > 1
        O_flat = self.O_predictor(hidden_state)  # (B, T, dag_depth * total_nodes)
        O = O_flat.view(
            B, T, self.dag_depth, self.total_nodes
        )  # (B, T, dag_depth, total_nodes)

        # Domain selector: 0=log, 1=linear
        G = torch.sigmoid(self.G_predictor(hidden_state))  # (B, T, dag_depth)

        # Apply STE sharpening if enabled for training
        if self.config.sharp_training:
            # Signs: STE to -1 or 1
            V_sign_hard = torch.where(
                V_sign >= 0,
                torch.tensor(1.0, device=V_sign.device),
                torch.tensor(-1.0, device=V_sign.device),
            )
            V_sign = V_sign_hard + (V_sign - V_sign.detach())

            # Operands: STE to nearest integer
            O = DAGExecutor.ste_round(O)

            # Gates: STE to 0 or 1
            G_hard = (G > 0.5).float()
            G = G_hard + (G - G.detach())  # G is already 0-1, so round to 0 or 1.

        return digit_logits, V_sign, O, G

    def to_text(self, digit_logits, V_sign, O, G, batch_idx=0, time_idx=0):
        """
            Convert predicted DAG tensors to human-readable text expression.

        Args:
                digit_logits: (B, T, num_initial_nodes, D, base) predicted digit logits
                V_sign: (B, T, total_nodes) predicted signs
                O: (B, T, dag_depth, total_nodes) predicted operand selectors
                G: (B, T, dag_depth) predicted domain gates
                batch_idx: Which batch element to convert (default: 0)
                time_idx: Which time step to convert (default: 0)

        Returns:
                str: Human-readable expression
        """
        # Extract single prediction from batch and time dimensions
        single_digit_logits = digit_logits[
            batch_idx, time_idx
        ]  # (num_initial_nodes, D, base)
        single_V_sign = V_sign[batch_idx, time_idx]  # (total_nodes,)
        single_O = O[batch_idx, time_idx]  # (dag_depth, total_nodes)
        single_G = G[batch_idx, time_idx]  # (dag_depth,)

        try:
            # Convert tensors to SymPy expression
            expr = tensor_to_expression(
                single_digit_logits,
                single_V_sign,
                single_O,
                single_G,
                self.max_digits,
                self.max_decimal_places,
            )

            # Convert to string
            return str(expr)

        except Exception as e:
            return f"[Error converting to expression: {e}]"


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
    max_digits: int = 4
    max_decimal_places: int = 4
    sharp_training: bool = False


class GPT(BaseGPTModel):
    """GPT model with DAG computation capability."""

    def __init__(self, config):
        super().__init__(config)

        if config.dag_depth > 0:
            # Use DAG system
            self.dag_predictor = DAGPlanPredictor(config)
            self.dag_executor = DAGExecutor(
                config.dag_depth, config.max_digits, config.max_decimal_places
            )
        else:
            self.dag_predictor = None
            self.dag_executor = None

    def forward(self, idx):
        """
        Forward pass with DAG computation.

        Args:
            idx: (B, T) input token indices

        Returns:
            (B, T) DAG execution results
        """
        # Get causal hidden states from transformer backbone
        hidden_states = self.forward_hidden(idx)  # (B, T, n_embd)

        if self.dag_predictor is not None:
            # Predict DAG tensors for each token position
            digit_logits, V_sign, O, G = self.dag_predictor(
                hidden_states
            )  # (B, T, ...)

            # Execute DAG - the executor handles digit conversion internally
            # Pass apply_ste based on config.sharp_training
            return self.dag_executor(
                digit_logits, V_sign, O, G, apply_ste=self.config.sharp_training
            )  # (B, T)
        else:
            # No DAG - this shouldn't happen in practice
            raise ValueError("DAG depth is 0, cannot perform DAG computation")
