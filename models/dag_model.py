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

from .base_model import BaseGPTModel

# Log-space arithmetic utilities
LOG_LIM = (
    100.0  # Bound on ln-magnitudes (increased to handle extreme large expressions)
)


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
        max_decimal_places: int,
        base: int = 10,
        temperature: float = 0.01,
    ) -> torch.Tensor:
        """
        Convert digit logits to V_mag tensor using soft/differentiable conversion.

        Args:
            digit_logits: (B, T, num_initial_nodes, D, base) - digit prediction logits
            max_digits: Maximum number of integer digits
            max_decimal_places: Maximum number of decimal places
            base: Numerical base (default 10)
            temperature: Temperature for softmax (lower = more discrete)

        Returns:
            V_mag: (B, T, num_initial_nodes) - magnitude values for initial nodes
        """
        B, T, num_initial_nodes, D, _ = digit_logits.shape

        # Use same dtype as input for consistency
        compute_dtype = digit_logits.dtype

        # Use low temperature to make softmax nearly one-hot for accuracy
        # This preserves gradients while minimizing conversion error
        digit_probs = torch.softmax(
            digit_logits / temperature, dim=-1
        )  # (B, T, num_initial_nodes, D, base)

        # Create digit value matrix: [0, 1, 2, ..., base-1]
        digit_values = torch.arange(
            base, dtype=compute_dtype, device=digit_logits.device
        )

        # Soft expected digit values: sum(prob_i * digit_i)
        expected_digits = (digit_probs * digit_values.view(1, 1, 1, 1, -1)).sum(
            dim=-1
        )  # (B, T, N, D)

        # Initialize output tensor
        V_mag = torch.zeros(
            B, T, num_initial_nodes, dtype=compute_dtype, device=digit_logits.device
        )

        for b in range(B):
            for t in range(T):
                for n in range(num_initial_nodes):
                    try:
                        # Integer part (positions 0 to max_digits-1)
                        int_value = 0.0
                        for d in range(max_digits):
                            digit_val = expected_digits[b, t, n, d]
                            # Validate digit value is reasonable
                            if (
                                not torch.isfinite(digit_val)
                                or digit_val < 0
                                or digit_val >= base
                            ):
                                raise ValueError(
                                    f"Invalid digit value {digit_val} at position {d} for node {n}"
                                )
                            int_value += digit_val * (base ** (max_digits - 1 - d))

                        # Fractional part (positions max_digits to D-1)
                        frac_value = 0.0
                        for d in range(max_digits, max_digits + max_decimal_places):
                            digit_val = expected_digits[b, t, n, d]
                            # Validate digit value is reasonable
                            if (
                                not torch.isfinite(digit_val)
                                or digit_val < 0
                                or digit_val >= base
                            ):
                                raise ValueError(
                                    f"Invalid digit value {digit_val} at position {d} for node {n}"
                                )
                            frac_value += digit_val * (base ** (max_digits - 1 - d))

                        # Combine and validate total
                        total_value = int_value + frac_value
                        if not torch.isfinite(total_value):
                            raise ValueError(
                                f"Digit-to-float conversion produced non-finite value: {total_value}"
                            )

                        # Clip to reasonable range to prevent inf/nan in operations
                        clipped_value = torch.clamp(total_value, min=1e-8, max=1e6)
                        V_mag[b, t, n] = clipped_value

                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to convert digits to magnitude for batch {b}, token {t}, node {n}: {e}"
                        ) from e

        return V_mag

    def forward(self, digit_logits, V_sign, O, G):
        """
            Execute DAG operations using tensor representation with digit prediction.

        Args:
                digit_logits: (B, T, num_initial_nodes, D, base) - digit predictions for initial nodes
                V_sign: (B, T, total_nodes) - signs of all nodes
                O: (B, T, dag_depth, total_nodes) - operand selection matrix
                G: (B, T, dag_depth) - domain selector (0=log, 1=linear)

        Returns:
                torch.Tensor: final result (B, T)
        """
        B, T = V_sign.shape[:2]

        # Store original dtype for final result
        original_dtype = V_sign.dtype

        # Cast to float64 for precise computation (except on MPS)
        compute_dtype = torch.float64 if V_sign.device.type != "mps" else torch.float32

        # Convert digit predictions to V_mag for initial nodes using shared method
        initial_V_mag = self.digits_to_vmag(
            digit_logits, self.max_digits, self.max_decimal_places, self.base
        )  # (B, T, num_initial_nodes)

        # Create full working tensors
        working_V_mag = torch.zeros(
            B, T, self.total_nodes, dtype=compute_dtype, device=V_sign.device
        )
        working_V_sign = V_sign.clone().to(compute_dtype)

        # Copy initial node magnitudes into working tensor (convert dtype if needed)
        working_V_mag[:, :, : self.num_initial_nodes] = initial_V_mag.to(compute_dtype)

        # Execute each intermediate computation step
        for step in range(self.num_intermediate_nodes):
            # Get operand selector and domain gate for this step
            O_step = O[:, :, step, :].to(compute_dtype)  # (B, T, total_nodes)
            G_step = G[:, :, step].unsqueeze(-1).to(compute_dtype)  # (B, T, 1)

            # Apply triangular mask to prevent using future intermediate results
            valid_positions = (
                self.num_initial_nodes + step
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

            # For magnitude: use safer clamping to prevent exp overflow
            linear_mag = torch.abs(R_mag)

            # For log domain, clamp R_mag before exp to prevent overflow
            R_mag_clamped = torch.clamp(
                R_mag, min=-LOG_LIM, max=LOG_LIM
            )  # Use LOG_LIM for proper range
            log_mag_result = torch.exp(R_mag_clamped)

            V_mag_new = G_step * linear_mag + (1 - G_step) * log_mag_result

            # Write result to the predetermined intermediate slot
            intermediate_idx = self.num_initial_nodes + step
            # Use scatter to avoid in-place operations that break gradients
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
        O_flat = self.O_predictor(hidden_state)  # (B, T, dag_depth * total_nodes)
        O = O_flat.view(
            B, T, self.dag_depth, self.total_nodes
        )  # (B, T, dag_depth, total_nodes)

        # Domain selector: 0=log, 1=linear
        G = torch.sigmoid(self.G_predictor(hidden_state))  # (B, T, dag_depth)

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
        from data.dagset.streaming import tensor_to_expression

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
            return self.dag_executor(digit_logits, V_sign, O, G)  # (B, T)
        else:
            # No DAG - this shouldn't happen in practice
            raise ValueError("DAG depth is 0, cannot perform DAG computation")
