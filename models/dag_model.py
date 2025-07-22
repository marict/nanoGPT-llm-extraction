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

import inspect
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from rff.layers import PositionalEncoding
from torch.nn import functional as F

from tensor_utils import index_copy_like

from .base_model import BaseGPTModel
from .components import Block, LayerNorm

LN10 = math.log(10.0)


# Log-space arithmetic utilities
LOG_LIM = 10.0  # Bound on log-magnitudes to avoid numerical instabilities
MIN_CLAMP = 1e-6


def _clip_log(log_t: torch.Tensor) -> torch.Tensor:
    """Symmetric tanh clipping for log magnitudes with minimum of -1."""
    return torch.tanh(log_t / LOG_LIM) * LOG_LIM


# Debug utility
ENABLE_DEBUG_NAN_CHECKS = os.getenv("DAGGPT_DEBUG_NANS", "0") == "1"
if ENABLE_DEBUG_NAN_CHECKS:
    print("dag_model: DEBUG_NAN_CHECKS is enabled")


def _debug_check(op_name: str, *tensors: torch.Tensor):
    """Check tensors for NaN/Inf values and print detailed debug info when found."""
    for idx, t in enumerate(tensors):
        if t is None:
            continue

        if torch.isfinite(t).all():
            continue

        nan_mask = torch.isnan(t)
        inf_mask = torch.isinf(t)
        nan_cnt = int(nan_mask.sum())
        inf_cnt = int(inf_mask.sum())

        print(
            f"[DEBUG] {op_name}: tensor#{idx} shape={tuple(t.shape)} dtype={t.dtype} "
            f"-> NaN: {nan_cnt}  Inf: {inf_cnt}"
        )

        finite_vals = t[torch.isfinite(t)].float()
        if finite_vals.numel() > 0:
            stats = (
                float(finite_vals.min()),
                float(finite_vals.mean()),
                float(finite_vals.max()),
            )
            print(f"           finite stats (min/mean/max): {stats}")

        # Show first 5 bad indices for debugging
        bad_idx = (~torch.isfinite(t)).nonzero(as_tuple=False)
        for j, coord in enumerate(bad_idx[:5]):
            coord_tuple = tuple(coord.tolist())
            bad_val = t[coord_tuple].item()
            print(f"           [{j}] index={coord_tuple} value={bad_val}")

        raise RuntimeError("NaN/Inf detected; see logs above for details")


def multiply_log_space(
    sx: torch.Tensor,
    lx: torch.Tensor,
    sy: torch.Tensor,
    ly: torch.Tensor,
    ignore_clip: bool = False,
):
    sgn_out = sx * sy
    log_out = lx + ly
    if not ignore_clip:
        log_out = _clip_log(log_out)
    if ENABLE_DEBUG_NAN_CHECKS:
        _debug_check("multiply_log_space", sgn_out, log_out)
    return sgn_out, log_out


def divide_log_space(
    sx: torch.Tensor,
    lx: torch.Tensor,
    sy: torch.Tensor,
    ly: torch.Tensor,
    ignore_clip: bool = False,
):
    sgn_out = sx * sy
    log_out = lx - ly
    if not ignore_clip:
        log_out = _clip_log(log_out)
    if ENABLE_DEBUG_NAN_CHECKS:
        _debug_check("divide_log_space", sgn_out, log_out)
    return sgn_out, log_out


# Helper when operands share the same sign
def _add_logs_same_sign(
    sgn: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor, ignore_clip: bool = False
):
    # ``torch.logaddexp`` assumes natural logarithm inputs. Our internal
    # representation uses log₁₀, so convert before/after to ensure numerical
    # correctness (resolves large discrepancies caught by SymPy equivalence
    # tests).

    # Convert to double precision for higher numerical accuracy before calling
    # torch.logaddexp which operates in the natural logarithm domain.  The
    # original implementation down-cast to ``float32`` for speed, but this
    # introduced small discrepancies (≈1e-5 relative) that can accumulate in
    # deep subtraction chains and trigger the SymPy equivalence assertion.
    # MPS backend (Apple Silicon) does not support ``float64``.  Detect this
    # case and fall back to ``float32`` to avoid runtime errors while still
    # using higher precision on CUDA/CPU devices that *do* support doubles.
    target_dtype = torch.float64 if lx.device.type != "mps" else torch.float32

    lx_n = lx.to(target_dtype) * LN10
    ly_n = ly.to(target_dtype) * LN10
    l_out_n = torch.logaddexp(lx_n, ly_n)
    l_out = (l_out_n / LN10).to(lx.dtype)
    if not ignore_clip:
        l_out = _clip_log(l_out)
    return sgn, l_out


def add_log_space(
    sx: torch.Tensor,
    lx: torch.Tensor,
    sy: torch.Tensor,
    ly: torch.Tensor,
    ignore_clip: bool = False,
):
    """Addition in log-space with sign handling (vectorised).
    Re-written without data-dependent Python control-flow so Torch Dynamo can
    capture it. All decisions are expressed with tensor operations.

    Args:
        sx: (B,T) tensor of signs
        lx: (B,T) tensor of log magnitudes
        sy: (B,T) tensor of signs
        ly: (B,T) tensor of log magnitudes
        ignore_clip: if True, skip clipping for testing
    Returns:
        s_out: (B,T) tensor of signs
        l_out: (B,T) tensor of log magnitudes
    """

    zero_x = sx == 0
    zero_y = sy == 0

    # Initialize outputs
    s_out = torch.zeros_like(sx)
    l_out = torch.zeros_like(lx)

    # Handle cases where only one operand is non-zero
    only_x = (~zero_x) & zero_y
    only_y = zero_x & (~zero_y)
    s_out = torch.where(only_x, sx, s_out)
    l_out = torch.where(only_x, lx, l_out)
    s_out = torch.where(only_y, sy, s_out)
    l_out = torch.where(only_y, ly, l_out)

    # Both operands non-zero
    both_nz = (~zero_x) & (~zero_y)
    same_sign = (sx * sy) > 0

    # Same sign: add magnitudes
    same_branch = both_nz & same_sign
    common_sgn = torch.sign(sx)
    s_same, l_same = _add_logs_same_sign(common_sgn, lx, ly, ignore_clip)
    s_out = torch.where(same_branch, s_same, s_out)
    l_out = torch.where(same_branch, l_same, l_out)

    # Opposite sign: subtract magnitudes
    opp_branch = both_nz & (~same_sign)
    bigger_is_x = lx >= ly
    big_log = torch.where(bigger_is_x, lx, ly)
    small_log = torch.where(bigger_is_x, ly, lx)
    big_sgn = torch.where(bigger_is_x, sx, sy)

    # Work in double precision to minimise error during calculations
    target_dtype = torch.float64 if lx.device.type != "mps" else torch.float32

    # Detect near-cancellation cases where log magnitudes are very close
    # In these cases, log-space arithmetic becomes numerically unstable
    log_diff = torch.abs(big_log - small_log)
    near_cancellation = log_diff < 1e-3  # Threshold for switching to linear arithmetic

    # For near-cancellation cases: use direct linear arithmetic
    # Convert back to linear space, compute difference, then back to log space
    big_val = torch.pow(
        torch.tensor(10.0, dtype=target_dtype, device=big_log.device),
        big_log.to(target_dtype),
    )
    small_val = torch.pow(
        torch.tensor(10.0, dtype=target_dtype, device=small_log.device),
        small_log.to(target_dtype),
    )

    # Compute linear difference with correct signs
    big_signed = torch.where(
        bigger_is_x, big_val * sx.to(target_dtype), big_val * sy.to(target_dtype)
    )
    small_signed = torch.where(
        bigger_is_x, small_val * sy.to(target_dtype), small_val * sx.to(target_dtype)
    )
    linear_result = big_signed + small_signed

    # Convert back to log space
    linear_abs = torch.abs(linear_result)
    linear_sgn = torch.sign(linear_result).to(sx.dtype)
    linear_log = torch.log10(linear_abs.clamp_min(MIN_CLAMP)).to(lx.dtype)

    # For non-near-cancellation cases: use original log-space approach
    delta_n = (small_log.to(target_dtype) - big_log.to(target_dtype)).clamp(
        max=-1e-3, min=-LOG_LIM
    ) * LN10
    diff_n = torch.log1p(-torch.exp(delta_n))  # natural log domain
    diff = (diff_n / LN10).to(lx.dtype)

    # Perfect cancellation case
    zero_res = small_log == big_log
    log_space_log = torch.where(zero_res, torch.zeros_like(big_log), big_log + diff)
    log_space_sgn = torch.where(zero_res, torch.zeros_like(big_sgn), big_sgn)

    # Choose between linear and log-space results based on near_cancellation flag
    new_sgn = torch.where(near_cancellation, linear_sgn, log_space_sgn)
    new_log = torch.where(near_cancellation, linear_log, log_space_log)

    s_out = torch.where(opp_branch, new_sgn, s_out)
    l_out = torch.where(opp_branch, new_log, l_out)

    # Final clipping and debug checks
    if not ignore_clip:
        l_out = _clip_log(l_out)
    if ENABLE_DEBUG_NAN_CHECKS:
        _debug_check("add_log_space", s_out, l_out)
    return s_out, l_out


def subtract_log_space(
    sx: torch.Tensor,
    lx: torch.Tensor,
    sy: torch.Tensor,
    ly: torch.Tensor,
    ignore_clip: bool = False,
):
    """Subtraction in log-space by negating the second operand."""
    s_out, l_out = add_log_space(sx, lx, -sy, ly, ignore_clip)
    if ENABLE_DEBUG_NAN_CHECKS:
        _debug_check("subtract_log_space", s_out, l_out)
    return s_out, l_out


def identity_log_space(  # Return the second operand, discard the first operand
    sx: torch.Tensor,  # Second operand
    lx: torch.Tensor,  # Second operand
    sy: torch.Tensor,  # First operand
    ly: torch.Tensor,  # First operand
    ignore_clip: bool = False,
):
    """Identity operation for stack execution."""
    return sx, lx


# For DAG execution, we may use a subset of the operations
OP_FUNCS = [
    add_log_space,
    subtract_log_space,
    multiply_log_space,
    divide_log_space,
    identity_log_space,
]
OP_NAMES = ["add", "subtract", "multiply", "divide", "identity"]

assert len(OP_FUNCS) == len(OP_NAMES), "OP_FUNCS and OP_NAMES must have the same length"


def safe_clamp(logits: torch.Tensor) -> torch.Tensor:
    if logits.dtype == torch.float16:
        max_val = 8.0
    elif logits.dtype == torch.bfloat16:
        max_val = 20.0
    else:
        max_val = 40.0
    return logits.clamp(-max_val, max_val)


class DAGPlanPredictor(nn.Module):
    """Predicts both initial values and operation choices for stack-based DAG execution."""

    def __init__(self, config):
        super().__init__()
        self.dag_depth = config.dag_depth
        init_ops_tau = 20.0
        # Learnable temperature (log τ) so model can adapt softmax sharpness during training
        # Parameterised in log-space and passed through softplus → guarantees τ > 0
        self.log_ops_tau = nn.Parameter(torch.log(torch.tensor(float(init_ops_tau))))

        # Learnable temperature for digit predictions to prevent plateauing
        init_digit_tau = 20.0  # Same as operations for consistency
        self.log_digit_tau = nn.Parameter(
            torch.log(torch.tensor(float(init_digit_tau)))
        )

        # Resolve operations from config
        self.op_names = config.op_names
        self.n_ops = len(self.op_names)
        self.num_scratch_nodes = config.dag_depth + 1
        self.n_embd = config.n_embd

        # Blocks for splitting hidden state
        self.initial_value_hidden = Block(config)
        self.dag_structure_hidden = Block(config)

        # Separate predictors for initial values and operations
        # Initial values predictor: hidden_state -> initial_values
        # Digits per number (fixed width)
        self.max_digits = config.max_digits
        self.max_decimal_places = config.max_decimal_places
        self.digits_per_number = self.max_digits + self.max_decimal_places

        # One sign + 10-way per digit slot per node
        initial_values_output_dim = self.num_scratch_nodes * (
            1 + self.digits_per_number * 10
        )
        self.initial_values_predictor = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, initial_values_output_dim),
        )

        # Bias digits toward zero at init (one time)
        with torch.no_grad():
            bias = self.initial_values_predictor[-1].bias
            digit_bias_start = self.num_scratch_nodes
            for node in range(self.num_scratch_nodes):
                for dig in range(self.digits_per_number):
                    offset = (
                        digit_bias_start + (node * self.digits_per_number + dig) * 10
                    )
                    bias[offset] = 3.0
                    bias[offset + 1 : offset + 10] = -3.0

            # Initialize weights for initial_values_predictor
            for layer in self.initial_values_predictor:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    # Bias was already set above for the last layer

        # Cross attention for operations: dag_structure_hidden attends to initial_value_hidden
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )

        # LayerNorm for stability
        self.cross_attn_ln = LayerNorm(config.n_embd, bias=config.bias)

        # Final projection to get operations output
        operations_output_dim = config.dag_depth * self.n_ops
        self.operations_projection = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, operations_output_dim),
        )

        # Initialize operations_projection weights and biases
        with torch.no_grad():
            # Set initial op softmax to nearly one-hot by biasing toward first op
            self.operations_projection[-1].bias.fill_(-4.0)

            # Initialize weights for operations_projection
            for layer in self.operations_projection:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    # Bias was already set above for the last layer

        # Store last predictions for logging
        self.last_operation_probs: torch.Tensor | None = None
        self.last_digit_logits: torch.Tensor | None = None

    def clear_cache(self) -> None:
        """Clear cached tensors to prevent memory leaks."""
        self.last_operation_probs = None
        self.last_digit_logits = None

    def forward(self, original_hidden: torch.Tensor):
        """
        ƒ both initial values and operation choices for stack-based execution.

        Args:
            original_hidden: (B, T, H) - original hidden states
        Returns:
                initial_sgn: (B, T, num_scratch_nodes) - initial signs
                initial_log: (B, T, num_scratch_nodes) - initial log magnitudes
                operation_probs: (B, T, dag_depth, n_ops) - probabilities for operations
        """
        # Clear previously cached tensors
        self.last_operation_probs = None

        B, T, H = original_hidden.shape

        # Split hidden state into initial value and dag structure components
        initial_value_hidden = self.initial_value_hidden(original_hidden)
        dag_structure_hidden = self.dag_structure_hidden(original_hidden)

        # Process initial values
        initial_value_flat = initial_value_hidden.reshape(B * T, H)
        initial_values_raw = self.initial_values_predictor(initial_value_flat)
        initial_values_raw = initial_values_raw.view(B, T, -1)

        # Process initial values
        slice_sign = self.num_scratch_nodes
        slice_digits = self.num_scratch_nodes * self.digits_per_number * 10

        digit_logits = initial_values_raw[..., slice_sign : slice_sign + slice_digits]

        # Ensure contiguous memory layout before reshaping.
        digit_logits = digit_logits.contiguous().view(
            B, T, self.num_scratch_nodes, self.digits_per_number, 10
        )

        # Store the last digit logits predicted for logging / compatibility
        self.last_digit_logits = digit_logits.clone()

        # Apply learnable temperature scaling to digit logits before softmax
        # Follow the same pattern as operation prediction for consistency
        digit_tau = (F.softplus(self.log_digit_tau) + 1e-4).to(digit_logits.dtype)
        digit_scale = math.sqrt(10)  # 10 digit classes, same pattern as ops
        digit_probs = F.softmax(
            digit_logits * digit_scale / digit_tau, dim=-1
        ).contiguous()
        sign_logits = initial_values_raw[..., :slice_sign]

        # Convert sign tensor in [-1,1]
        initial_sgn = torch.tanh(sign_logits)

        # Process operations:
        # Cross attention: dag_structure_hidden attends to initial_value_hidden
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=original_hidden.device), diagonal=1
        )
        cross_output, _ = self.cross_attn(
            query=dag_structure_hidden,
            key=initial_value_hidden,
            value=initial_value_hidden,
            attn_mask=causal_mask,
            is_causal=True,
        )

        # Apply LayerNorm for stability
        cross_output = self.cross_attn_ln(cross_output)

        # Project to operations output
        cross_output_flat = cross_output.reshape(B * T, H)
        operation_logits_raw = self.operations_projection(cross_output_flat)
        operation_logits_raw = operation_logits_raw.view(
            B, T, -1
        )  # (B, T, dag_depth * n_ops)

        operation_logits = operation_logits_raw.view(B, T, self.dag_depth, self.n_ops)

        # Clamp logits before softmax for numerical stability
        operation_logits = safe_clamp(operation_logits)

        # Standard softmax with *learnable* temperature (subset size for stability)
        tau = (F.softplus(self.log_ops_tau) + 1e-4).to(
            operation_logits.dtype
        )  # ensure τ>0
        scale = math.sqrt(self.n_ops)
        operation_probs_subset = F.softmax(
            operation_logits * scale / tau, dim=-1
        )  # (B,T,depth,subset)
        # Ensure dtype matches operation_logits to avoid index_copy_ dtype mismatches
        operation_probs_subset = operation_probs_subset.to(operation_logits.dtype)

        # Map subset probabilities back to full OP_NAMES length
        if self.n_ops == len(OP_NAMES):
            # No subset → return directly to avoid extra alloc
            operation_probs = operation_probs_subset
        else:
            full_ops = operation_logits.new_zeros(B, T, self.dag_depth, len(OP_NAMES))
            idx = torch.tensor(
                [OP_NAMES.index(op) for op in self.op_names],
                device=operation_logits.device,
            )
            # Scatter along last dim
            full_ops.index_copy_(-1, idx, operation_probs_subset)
            operation_probs = full_ops

        # Cache operation probs for logging
        self.last_operation_probs = operation_probs
        return initial_sgn, digit_probs, operation_probs


def apply_op(
    s1: torch.Tensor,
    l1: torch.Tensor,
    s2: torch.Tensor,
    l2: torch.Tensor,
    op_probs: torch.Tensor,
    *,
    ignore_clip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a predicted op to log-space values using soft selection over ops.

    Args:
        s1, l1, s2, l2: operands in sign/log¹⁰ form
        op_probs: probabilities over operations (… , n_ops)
        ignore_clip: if True, disable log-magnitude clipping inside operations
    """
    # Apply all operations with the specified clipping behaviour
    add_sgn, add_log = add_log_space(s1, l1, s2, l2, ignore_clip)
    sub_sgn, sub_log = subtract_log_space(s1, l1, s2, l2, ignore_clip)
    mul_sgn, mul_log = multiply_log_space(s1, l1, s2, l2, ignore_clip)
    div_sgn, div_log = divide_log_space(s1, l1, s2, l2, ignore_clip)
    id_sgn, id_log = identity_log_space(s1, l1, s2, l2, ignore_clip)

    # Stack results
    ops_sgn = torch.stack(
        [add_sgn, sub_sgn, mul_sgn, div_sgn, id_sgn], dim=-1
    )  # (B, T, 5)
    ops_log = torch.stack(
        [add_log, sub_log, mul_log, div_log, id_log], dim=-1
    )  # (B, T, 5)

    # Apply soft selection using probabilities
    result_sgn = (ops_sgn * op_probs).sum(dim=-1)  # (B, T)
    result_log = (ops_log * op_probs).sum(dim=-1)  # (B, T)

    return result_sgn, result_log


def _debug_apply_op(op_name: str, second: float, top: float) -> float:
    """Apply an operation to two floats and return the result."""
    if op_name == "add":
        return second + top
    if op_name == "subtract":
        return second - top
    if op_name == "multiply":
        return second * top
    if op_name == "divide":
        return second / top
    if op_name == "identity":
        return second


def execute_stack(
    initial_sgn: torch.Tensor,  # (B,T,N) values in [-1,1] (tanh output)
    digit_probs: torch.Tensor,  # (B,T,N,D,10) – probabilities (softmaxed)
    ops: torch.Tensor,  # (B,T,depth,n_ops) operation probabilities per step
    *,
    max_digits: int,
    max_decimal_places: int,
    ignore_clip: bool = False,
    _print_exec_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute a differentiable stack-based DAG computation.

    This function expects the **raw** predictor outputs: continuous sign values
    in [-1,1] (after `tanh`) and digit probability tensors.

    Args:
        initial_sgn:  (B,T,N) continuous sign values (range [-1,1])
        digit_probs: (B,T,N,D,10) probabilities over each digit slot
        ops:         (B,T,depth,n_ops) operation probabilities per step
        max_digits:  number of integer digit slots (D_int)
        max_decimal_places: number of fractional digit slots (D_frac)

    Returns:
        final_sgn, final_log: (B,T) tensors representing the output scalar in
        signed–log₁₀ space.
    """

    if max_digits + max_decimal_places != digit_probs.shape[3]:
        raise RuntimeError(
            f"max_digits + max_decimal_places ({max_digits + max_decimal_places}) must match digit_probs.shape[3]: ({digit_probs.shape[3]})"
        )

    if max_digits > 10:
        raise RuntimeError(
            f"max_digits ({max_digits}) must be <= 10 because numbers larger than 10¹⁰ would exceed "
            f"LOG_LIM={LOG_LIM} and be clipped, making sympy comparisons invalid"
        )

    # ------------------------------------------------------------------
    # Digit probabilities tensor must be 5-D: (B, T, N, D, 10)
    if digit_probs.dim() != 5:
        raise RuntimeError(
            "digit_probs tensor must have shape (B, T, N, D, 10); got dim="
            f"{digit_probs.dim()}"
        )

    # Expected digit value per slot to build absolute magnitude
    digits_values = torch.arange(10, device=digit_probs.device, dtype=digit_probs.dtype)
    expected_digits = (digit_probs * digits_values).sum(-1)  # (B,T,N,D)

    # Convert expected digits to log magnitude
    # for operations.
    powers = torch.arange(
        max_digits + max_decimal_places - 1,
        -1,
        -1,
        device=digit_probs.device,
        dtype=digit_probs.dtype,
    )
    # Assemble integer value from digits and powers
    raw_int_value = (expected_digits * (10.0**powers)).sum(-1)  # (B,T,N)
    # Separate fixed scale (constant)
    scale = max_decimal_places
    raw_int_value = raw_int_value.clamp_min(MIN_CLAMP)
    initial_log = torch.log10(raw_int_value) - scale

    B, T, num_initial = initial_sgn.shape
    depth = ops.shape[2]

    # Pre-allocate buffers
    buffer_sgn = torch.zeros(
        B,
        T,
        num_initial,
        device=initial_sgn.device,
        dtype=initial_sgn.dtype,
    )
    buffer_log = torch.zeros(
        B,
        T,
        num_initial,
        device=initial_log.device,
        dtype=initial_log.dtype,
    )

    buffer_sgn = buffer_sgn + initial_sgn
    buffer_log = buffer_log + initial_log

    current_size = num_initial

    # Helper function for debug output
    def _log_to_value(sgn: torch.Tensor, log: torch.Tensor) -> float:
        """Convert sign/log representation back to readable value."""
        if sgn.numel() == 1 and log.numel() == 1:
            return float(sgn.item() * (10 ** log.item()))
        return "BATCH_DATA"

    if _print_exec_intermediates:
        print("\n=== EXECUTE_STACK DEBUG ===")
        print(f"Initial stack size: {current_size}")
        print(f"Operations depth: {depth}")
        for i in range(current_size):
            val = _log_to_value(buffer_sgn[0, 0, i], buffer_log[0, 0, i])
            print(f"  Stack[{i}]: {val}")
        print()

    for step in range(depth):
        if current_size < 2:
            raise RuntimeError(
                f"Stack underflow at step {step}: only {current_size} elements remaining, need at least 2"
            )

        top_sgn = buffer_sgn[..., current_size - 1]
        second_sgn = buffer_sgn[..., current_size - 2]
        top_log = buffer_log[..., current_size - 1]
        second_log = buffer_log[..., current_size - 2]

        if _print_exec_intermediates:
            # Show operation details
            op_probs = ops[0, 0, depth - 1 - step]  # Get operation probabilities
            max_op_idx = torch.argmax(op_probs).item()
            op_name = (
                OP_NAMES[max_op_idx]
                if max_op_idx < len(OP_NAMES)
                else f"unknown_{max_op_idx}"
            )
            op_prob = op_probs[max_op_idx].item()

            second_val = _log_to_value(second_sgn[0, 0], second_log[0, 0])
            top_val = _log_to_value(top_sgn[0, 0], top_log[0, 0])

            print(f"Step {step}: Applying '{op_name}' (prob={op_prob:.4f})")
            print(f"Operands: second={second_val}, top={top_val}")
            print(f"Stack indices: [{current_size-2}] op [{current_size-1}]")
            print(f"Correct Result: {_debug_apply_op(op_name, second_val, top_val)}")

        # Apply the operation in Reverse Polish Notation
        # Ex. ['add','identity'] will apply identity operation first, and then add
        result_sgn, result_log = apply_op(
            second_sgn,
            second_log,
            top_sgn,
            top_log,
            ops[:, :, depth - 1 - step],
            ignore_clip=ignore_clip,
        )

        if _print_exec_intermediates:
            result_val = _log_to_value(result_sgn[0, 0], result_log[0, 0])
            print(f"  Result: {result_val}")

        second_idx = current_size - 2
        idx = torch.tensor([second_idx], device=buffer_sgn.device)
        buffer_sgn = index_copy_like(buffer_sgn, -1, idx, result_sgn)
        buffer_log = index_copy_like(buffer_log, -1, idx, result_log)

        current_size -= 1

        if _print_exec_intermediates:
            print(f"  New stack size: {current_size}")
            for i in range(current_size):
                val = _log_to_value(buffer_sgn[0, 0, i], buffer_log[0, 0, i])
                print(f"    Stack[{i}]: {val}")
            print()

    if _print_exec_intermediates:
        final_val = _log_to_value(buffer_sgn[0, 0, 0], buffer_log[0, 0, 0])
        print(f"FINAL RESULT: {final_val}")
        print(f"  Final sign: {buffer_sgn[0, 0, 0].item()}")
        print(f"  Final log:  {buffer_log[0, 0, 0].item()}")
        print("=== END EXECUTE_STACK DEBUG ===\n")

    return buffer_sgn[..., 0], buffer_log[..., 0]


class ScalarToEmbed(nn.Module):
    """Map sign & log magnitude to embedding using Fourier bases."""

    def __init__(self, hidden: int, sigma=1.0, feat_dim: int = 32):
        super().__init__()
        self.ff = PositionalEncoding(sigma, feat_dim)
        self.proj = nn.Linear(2 * feat_dim + 2, hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        sign = feat[..., 0:1]
        logmag = feat[..., 1:2]
        trig = self.ff(logmag)
        concat = torch.cat((sign, logmag, trig), dim=-1)
        return self.proj(concat)


def verify_ops(op_names: list[str]):
    """Verify that the operations are valid."""
    for op in op_names:
        if op not in OP_NAMES:
            raise ValueError(f"Invalid operation: {op}")


# ---------------------------------------------------------------------------
# DAG computation
class DifferentiableDAG(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.hidden_dim = config.n_embd
        self.dag_depth = config.dag_depth
        self.op_names = config.op_names
        self.ops = [OP_FUNCS[OP_NAMES.index(op)] for op in self.op_names]

        # Always use dag_depth + 1 initial values for optimal stack-based computation
        self.num_scratch_nodes = config.dag_depth + 1

        # Validate configuration at initialization time
        if self.dag_depth < 1:
            raise ValueError(
                f"DAG depth must be at least 1 to ensure ≥2 initial values, got {self.dag_depth}. "
                f"Use dag_depth=0 for standard GPT or dag_depth≥1 for DAG-augmented GPT."
            )

        self.post_dag = Block(config)

        # Initialize unified plan predictor (predicts both initial values and operations)
        self.plan_predictor = DAGPlanPredictor(config)
        self.scalar_to_embed = ScalarToEmbed(config.n_embd)

        self.final_hidden = None
        self.final_values = None

    def clear_cache(self) -> None:
        """Clear cached tensors to prevent memory leaks."""
        self.final_hidden = None
        self.final_values = None
        # Also clear plan predictor cache
        self.plan_predictor.clear_cache()

    def forward(
        self,
        original_hidden: torch.Tensor,  # (B, T, H)
    ):
        # Clear previously cached tensors
        self.final_hidden = None
        self.final_values = None

        # Generate unified plan (initial values + operations)
        initial_sgn, digit_probs, operation_probs = self.plan_predictor(original_hidden)

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("initial_values_sgn", initial_sgn)

        final_sgn, final_log = execute_stack(
            initial_sgn,
            digit_probs,
            operation_probs,
            max_digits=self.plan_predictor.max_digits,
            max_decimal_places=self.plan_predictor.max_decimal_places,
        )

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("final_values", final_sgn, final_log)

        # Convert result to embedding
        final_feat = torch.stack((final_sgn, final_log), dim=-1)
        final_hidden = self.scalar_to_embed(final_feat)

        # Apply post-processing
        final_hidden = self.post_dag(final_hidden)

        # Cache for logging
        self.final_hidden = final_hidden
        # Convert digit_probs to magnitudes for logging purposes
        digits_vals = (
            digit_probs
            * torch.arange(10, device=digit_probs.device, dtype=digit_probs.dtype)
        ).sum(-1)
        int_weights = (
            10
            ** torch.arange(
                self.plan_predictor.max_digits - 1,
                -1,
                -1,
                device=digit_probs.device,
            )
        ).to(digit_probs.dtype)
        frac_weights = (
            10
            ** torch.arange(
                -1,
                -self.plan_predictor.max_decimal_places - 1,
                -1,
                device=digit_probs.device,
            )
        ).to(digit_probs.dtype)
        weights = torch.cat((int_weights, frac_weights))
        value_abs = (digits_vals * weights).sum(-1)
        initial_log = torch.log(value_abs.clamp_min(1e-6)) / math.log(10.0)

        self.final_values = (
            (initial_sgn * 10**initial_log).permute(0, 2, 1).contiguous()
        )

        return final_hidden


# GPT configuration
@dataclass
class GPTConfig:
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
    dag_depth: int = (
        4  # 0 = standard GPT, >0 = DAG-augmented GPT (minimum 1 for ≥2 initial values)
    )
    # Operation names the model should predict; defaults to the full set.
    op_names: list[str] = field(default_factory=lambda: OP_NAMES.copy())

    # Digit configuration for initial value prediction
    max_digits: int = 4  # Integer digits
    max_decimal_places: int = 6  # Fractional digits


# Main GPT model
class GPT(BaseGPTModel):
    """GPT Language Model with optional DAG augmentation.

    When dag_depth=0, behaves as a standard GPT model.
    When dag_depth>0, uses differentiable DAG for enhanced reasoning.
    """

    def __init__(self, config: GPTConfig):
        assert config.vocab_size is not None
        assert config.block_size is not None

        # Validate dag_depth
        if config.dag_depth < 0:
            raise ValueError(
                f"DAG depth cannot be negative, got {config.dag_depth}. "
                f"Use dag_depth=0 for standard GPT or dag_depth≥1 for DAG-augmented GPT."
            )

        # Call base class constructor which handles transformer creation and weight init
        super().__init__(config)

        # Create language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # DAG-specific components (only if dag_depth > 0)
        if config.dag_depth > 0:
            self.dag = DifferentiableDAG(config)

            # Initialise gate closed (~0.12 after sigmoid)
            self.gate_w_d = nn.Parameter(torch.full((config.n_embd,), -2.0))
            self.gate_w_o = nn.Parameter(torch.full((config.n_embd,), -2.0))
            # Bias so gate can open even if means are near zero
            self.gate_b = nn.Parameter(torch.tensor(-2.0))

    def clear_model_cache(self) -> None:
        """Clear all cached tensors to prevent memory leaks."""
        # Clear stored tensors from forward pass
        if hasattr(self, "last_original_hidden"):
            self.last_original_hidden = None
        if hasattr(self, "last_mixed_hidden"):
            self.last_mixed_hidden = None
        if hasattr(self, "final_values"):
            self.final_values = None
        if hasattr(self, "last_gate"):
            self.last_gate = None

        # Clear DAG cache if present
        if hasattr(self, "dag"):
            self.dag.clear_cache()

    def _compute_loss(self, logits, targets):
        """Utility to compute the cross-entropy loss."""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

    def forward(self, idx, targets=None):
        _, T = idx.shape
        device = idx.device
        assert T <= self.config.block_size

        # GPT backbone
        pos = torch.arange(T, device=device)
        emb = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(emb)
        for blk in self.transformer.h:
            x = blk(x)
        hidden = self.transformer.ln_f(x)

        # Standard GPT mode (no DAG augmentation)
        if self.config.dag_depth == 0:
            if targets is not None:
                logits = self.lm_head(hidden)
                loss = self._compute_loss(logits, targets)
            else:
                logits = self.lm_head(hidden[:, [-1], :])
                loss = None

            return logits, loss

        # DAG augmentation mode
        original_hidden = hidden.clone()

        # Store for logging
        self.last_original_hidden = original_hidden

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("original_hidden", original_hidden)

        # Run DAG processing
        dag_hidden = self.dag(original_hidden)

        # Store for logging
        self.final_values = self.dag.final_values

        # Per-token gating: mix original and DAG hidden states
        gate_logits = (
            (dag_hidden * self.gate_w_d).mean(-1, keepdim=True)
            + (original_hidden * self.gate_w_o).mean(-1, keepdim=True)
            + self.gate_b
        )
        gate = torch.sigmoid(gate_logits)

        # Store for logging
        self.last_gate = gate

        mixed_hidden = original_hidden + gate * dag_hidden

        # Store for logging
        self.last_mixed_hidden = mixed_hidden

        logits = self.lm_head(mixed_hidden)
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)
            if torch.is_grad_enabled():
                assert loss.requires_grad, "Loss does not require grad"
                assert loss.grad_fn is not None, "Loss has no grad_fn"

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # Model configurations
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # Create model and load weights
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy weights with proper handling of Conv1D layers
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text by sampling from the model."""
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Get logits and apply temperature
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Normalize and sample
            logits_std = logits.std(dim=-1, keepdim=True).clamp_min(1e-3)
            logits = logits / logits_std
            probs = F.softmax(logits, dim=-1)

            # Handle edge cases
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            else:
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
