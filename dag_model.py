"""
Full definition of a GPT Language Model with optional DAG augmentation.
When dag_depth=0, behaves as a standard GPT model.
When dag_depth>0, uses differentiable DAG for enhanced reasoning.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from rff.layers import PositionalEncoding
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Log-space arithmetic utilities
LOG_LIM = 10.0  # Bound on log-magnitudes to avoid numerical instabilities
GRAD_CAP = 0.3  # try 0.1–10.0 depending on bfloat16/float16 range


def _clip_log(log_t: torch.Tensor) -> torch.Tensor:
    """Symmetric tanh clipping for log magnitudes."""
    return torch.tanh(log_t / LOG_LIM) * LOG_LIM


def _rms_rescale(prev_logs: torch.Tensor, new_log: torch.Tensor) -> torch.Tensor:
    """Return a RMS-rescaled version of *new_log* so that, when appended to
    *prev_logs*, the running RMS equals ``LOG_LIM``.

    No in-place mutation → avoids autograd versioning issues.

    Args:
        prev_logs: (B,T,S) tensor with existing log magnitudes.
        new_log:   (B,T)   tensor – the candidate log magnitude to append.
    Returns:
        new_log_scaled: (B,T) tensor, rescaled.
    """
    slice_ = torch.cat((prev_logs, new_log.unsqueeze(-1)), dim=-1)
    rms = torch.sqrt((slice_**2).mean(dim=-1, keepdim=True) + 1e-6)
    scale = (LOG_LIM / rms).clamp(max=1.0)
    return new_log * scale.squeeze(-1)


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
    # Perform high-precision accumulation to avoid bf16 overflows
    lx32, ly32 = lx.float(), ly.float()
    l_out = torch.logaddexp(lx32, ly32).to(lx.dtype)
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

    # Ensure delta < -1e-3 to keep gradients bounded
    delta32 = (small_log - big_log).float().clamp(max=-1e-3, min=-LOG_LIM)
    diff32 = torch.log1p(-torch.exp(delta32))  # safe in fp32
    diff = diff32.to(lx.dtype)

    # Perfect cancellation case
    zero_res = small_log == big_log
    new_log = torch.where(zero_res, torch.zeros_like(big_log), big_log + diff)
    new_sgn = torch.where(zero_res, torch.zeros_like(big_sgn), big_sgn)

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


def identity_log_space(
    sx: torch.Tensor, lx: torch.Tensor, *_, ignore_clip: bool = False
):
    return sx, lx


op_funcs = [
    add_log_space,
    subtract_log_space,
    multiply_log_space,
    divide_log_space,
    identity_log_space,
]
op_names = ["add", "subtract", "multiply", "divide", "identity"]


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

    def __init__(self, config, temperature: float = 10):
        super().__init__()
        self.dag_depth = config.dag_depth
        self.temperature = temperature
        self.n_ops = len(op_funcs)  # Number of operations
        self.num_scratch_nodes = config.dag_depth + 1
        self.n_embd = config.n_embd

        # Separate predictors for initial values and operations
        # Initial values predictor: hidden_state -> initial_values
        initial_values_output_dim = 2 * self.num_scratch_nodes
        self.initial_values_predictor = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, initial_values_output_dim),
        )

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

        # Store last predictions for logging
        self.last_operation_probs: torch.Tensor | None = None
        self.mag_logits: torch.Tensor | None = None  # for logging

    def forward(
        self, initial_value_hidden: torch.Tensor, dag_structure_hidden: torch.Tensor
    ):
        """
        Predict both initial values and operation choices for stack-based execution.

        Args:
            initial_value_hidden: (B, T, H) - hidden states for initial values
            dag_structure_hidden: (B, T, H) - hidden states for operations
        Returns:
            initial_sgn: (B, T, num_scratch_nodes) - initial signs
            initial_log: (B, T, num_scratch_nodes) - initial log magnitudes
            operation_probs: (B, T, dag_depth, n_ops) - probabilities for operations
        """
        B, T, H = initial_value_hidden.shape

        # Process initial values
        initial_value_flat = initial_value_hidden.reshape(B * T, H)
        initial_values_raw = self.initial_values_predictor(initial_value_flat)
        initial_values_raw = initial_values_raw.view(
            B, T, -1
        )  # (B, T, 2 * num_scratch_nodes)

        # Process operations using cross attention
        # Cross attention: dag_structure_hidden attends to initial_value_hidden
        cross_output, _ = self.cross_attn(
            query=dag_structure_hidden,
            key=initial_value_hidden,
            value=initial_value_hidden,
        )

        # Apply LayerNorm for stability
        cross_output = self.cross_attn_ln(cross_output)

        # Project to operations output
        cross_output_flat = cross_output.reshape(B * T, H)
        operation_logits_raw = self.operations_projection(cross_output_flat)
        operation_logits_raw = operation_logits_raw.view(
            B, T, -1
        )  # (B, T, dag_depth * n_ops)

        # Process initial values
        sign_logits = initial_values_raw[..., : self.num_scratch_nodes]
        mag_logits = initial_values_raw[..., self.num_scratch_nodes :]

        # Save for logging
        self.mag_logits = mag_logits

        # Convert to sign and log magnitude format
        initial_sgn = torch.tanh(sign_logits)
        initial_log = torch.abs(torch.tanh(mag_logits)) * LOG_LIM

        # Process operation logits
        operation_logits = operation_logits_raw.view(B, T, self.dag_depth, self.n_ops)

        # Clamp logits before softmax for numerical stability
        operation_logits = safe_clamp(operation_logits)

        # Standard softmax with temperature scaling
        scale = math.sqrt(self.n_ops)
        operation_probs = F.softmax(operation_logits * scale / self.temperature, dim=-1)

        # Cache for logging
        self.last_operation_probs = operation_probs

        return initial_sgn, initial_log, operation_probs


def apply_log_op(
    s1: torch.Tensor,
    l1: torch.Tensor,
    s2: torch.Tensor,
    l2: torch.Tensor,
    op_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a predicted op to log-space values using soft selection over ops."""
    # Apply all operations
    add_sgn, add_log = add_log_space(s1, l1, s2, l2)
    sub_sgn, sub_log = subtract_log_space(s1, l1, s2, l2)
    mul_sgn, mul_log = multiply_log_space(s1, l1, s2, l2)
    div_sgn, div_log = divide_log_space(s1, l1, s2, l2)
    id_sgn, id_log = identity_log_space(s1, l1)

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


def stack_based_execution_buffered(
    initial_values_sgn: torch.Tensor,
    initial_values_log: torch.Tensor,
    ops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute stack-based DAG computation using pre-allocated buffers.

    This avoids tensor concatenations at each step, improving memory efficiency.
    Uses careful indexing to preserve autograd compatibility.

    Args:
        initial_values_sgn: (B, T, num_initial) - initial stack signs
        initial_values_log: (B, T, num_initial) - initial stack log magnitudes
        ops: (B, T, depth, n_ops) - operation probabilities for each step

    Returns:
        final_sgn: (B, T) - final sign
        final_log: (B, T) - final log magnitude
    """
    B, T, num_initial = initial_values_sgn.shape
    depth = ops.shape[2]

    # Pre-allocate buffers for the entire computation
    # We need at most num_initial slots (starts with num_initial, reduces by 1 each step)
    buffer_sgn = torch.zeros(
        B,
        T,
        num_initial,
        device=initial_values_sgn.device,
        dtype=initial_values_sgn.dtype,
    )
    buffer_log = torch.zeros(
        B,
        T,
        num_initial,
        device=initial_values_log.device,
        dtype=initial_values_log.dtype,
    )

    # Initialize buffers with initial values
    buffer_sgn = buffer_sgn + initial_values_sgn  # Use addition to preserve gradients
    buffer_log = buffer_log + initial_values_log

    current_size = num_initial

    for step in range(depth):
        # Runtime check: ensure we have at least 2 elements to operate on
        if current_size < 2:
            raise RuntimeError(
                f"Stack underflow at step {step}: only {current_size} elements remaining, need at least 2"
            )

        # Extract top two elements (last two in current active region)
        top_sgn = buffer_sgn[..., current_size - 1]  # (B, T)
        second_sgn = buffer_sgn[..., current_size - 2]  # (B, T)
        top_log = buffer_log[..., current_size - 1]  # (B, T)
        second_log = buffer_log[..., current_size - 2]  # (B, T)

        # Apply operation
        result_sgn, result_log = apply_log_op(
            second_sgn, second_log, top_sgn, top_log, ops[:, :, step]
        )

        # Apply RMS rescaling to keep magnitudes bounded
        if current_size > 1:
            prev_logs_slice = buffer_log[
                ..., : current_size - 1
            ]  # (B, T, current_size-1)
            result_log = _rms_rescale(prev_logs_slice, result_log)

        # Update buffer: replace second-to-last element with result
        # Use advanced indexing to avoid in-place operations that break autograd
        new_buffer_sgn = buffer_sgn.clone()
        new_buffer_log = buffer_log.clone()

        # Update the element at position current_size-2 with the result
        new_buffer_sgn[..., current_size - 2] = result_sgn
        new_buffer_log[..., current_size - 2] = result_log

        # Update buffers (safe for autograd)
        buffer_sgn = new_buffer_sgn
        buffer_log = new_buffer_log

        # Reduce active size
        current_size -= 1

    # Return final result (should be at position 0)
    return buffer_sgn[..., 0], buffer_log[..., 0]


def stack_based_execution(
    initial_values_sgn: torch.Tensor,
    initial_values_log: torch.Tensor,
    ops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute stack-based DAG computation.

    Args:
        initial_values_sgn: (B, T, num_initial) - initial stack signs
        initial_values_log: (B, T, num_initial) - initial stack log magnitudes
        ops: (B, T, depth, n_ops) - operation probabilities for each step

    Returns:
        final_sgn: (B, T) - final sign
        final_log: (B, T) - final log magnitude
    """
    # Use the buffered implementation for better performance
    return stack_based_execution_buffered(initial_values_sgn, initial_values_log, ops)


def stack_based_execution_original(
    initial_values_sgn: torch.Tensor,
    initial_values_log: torch.Tensor,
    ops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Original stack-based execution (kept for testing/fallback).

    Args:
        initial_values_sgn: (B, T, num_initial) - initial stack signs
        initial_values_log: (B, T, num_initial) - initial stack log magnitudes
        ops: (B, T, depth, n_ops) - operation probabilities for each step

    Returns:
        final_sgn: (B, T) - final sign
        final_log: (B, T) - final log magnitude
    """
    B, T, num_initial = initial_values_sgn.shape
    depth = ops.shape[2]

    # Note: Configuration validation moved to DifferentiableDAG.__init__

    # Keep track of current values at each position
    # We'll build new tensors at each step to avoid in-place modifications
    current_sgn = initial_values_sgn  # (B, T, num_initial)
    current_log = initial_values_log  # (B, T, num_initial)

    stack_size = num_initial

    for step in range(depth):
        # Runtime check: ensure we have at least 2 elements to operate on
        if stack_size < 2:
            raise RuntimeError(
                f"Stack underflow at step {step}: only {stack_size} elements remaining, need at least 2"
            )

        # Pop top two elements (take last two in stack)
        top_sgn = current_sgn[..., stack_size - 1]  # (B, T)
        second_sgn = current_sgn[..., stack_size - 2]  # (B, T)
        top_log = current_log[..., stack_size - 1]  # (B, T)
        second_log = current_log[..., stack_size - 2]  # (B, T)

        # Apply operation
        result_sgn, result_log = apply_log_op(
            second_sgn, second_log, top_sgn, top_log, ops[:, :, step]
        )

        # Apply RMS rescaling to keep magnitudes bounded
        prev_logs_slice = current_log[..., : stack_size - 1]  # (B, T, stack_size-1)
        result_log = _rms_rescale(prev_logs_slice, result_log)

        # Create new tensors for the updated stack
        # Keep the elements before the second-to-top position, update the second-to-top with result
        if stack_size > 2:
            # Keep elements [0, ..., stack_size-3], then result at stack_size-2
            new_sgn = torch.cat(
                [
                    current_sgn[..., : stack_size - 2],  # Elements before second-to-top
                    result_sgn.unsqueeze(-1),  # New result
                ],
                dim=-1,
            )
            new_log = torch.cat(
                [
                    current_log[..., : stack_size - 2],  # Elements before second-to-top
                    result_log.unsqueeze(-1),  # New result
                ],
                dim=-1,
            )
        else:
            # Only two elements, result becomes the only element
            new_sgn = result_sgn.unsqueeze(-1)
            new_log = result_log.unsqueeze(-1)

        current_sgn = new_sgn
        current_log = new_log
        stack_size -= 1

    # Return final result (should be at position 0)
    return current_sgn[..., 0], current_log[..., 0]


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

        # Always use dag_depth + 1 initial values for optimal stack-based computation
        self.num_scratch_nodes = config.dag_depth + 1

        self.temperature = config.softmax_temperature

        # Validate configuration at initialization time
        if self.dag_depth < 1:
            raise ValueError(
                f"DAG depth must be at least 1 to ensure ≥2 initial values, got {self.dag_depth}. "
                f"Use dag_depth=0 for standard GPT or dag_depth≥1 for DAG-augmented GPT."
            )

        # Validate we have enough initial values for the requested depth
        if self.dag_depth >= self.num_scratch_nodes:
            raise ValueError(
                f"Cannot execute {self.dag_depth} DAG steps with only {self.num_scratch_nodes} initial values. "
                f"Each step consumes one value, need at least {self.dag_depth + 1} initial values."
            )

        self.initial_value_hidden = Block(config)
        self.dag_structure_hidden = Block(config)
        self.post_dag = Block(config)

        # Initialize unified plan predictor (predicts both initial values and operations)
        self.plan_predictor = DAGPlanPredictor(config, self.temperature)

        self.scalar_to_embed = ScalarToEmbed(config.n_embd)

    def forward(
        self,
        original_hidden: torch.Tensor,  # (B, T, H)
    ):
        # Pre-process hidden states
        initial_value_hidden = self.initial_value_hidden(original_hidden)
        dag_structure_hidden = self.dag_structure_hidden(original_hidden)

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("dag_structure_hidden", dag_structure_hidden)

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("initial_value_hidden", initial_value_hidden)

        # Generate unified plan (initial values + operations)
        initial_sgn, initial_log, operation_probs = self.plan_predictor(
            initial_value_hidden, dag_structure_hidden
        )

        if ENABLE_DEBUG_NAN_CHECKS:
            _debug_check("initial_values", initial_sgn, initial_log)

        # Execute stack-based computation
        final_sgn, final_log = stack_based_execution(
            initial_sgn, initial_log, operation_probs
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
        self.final_values = (initial_sgn * initial_log).permute(0, 2, 1).contiguous()

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
    softmax_temperature: float = 20.0


# Main GPT model
class GPT(nn.Module):
    """GPT Language Model with optional DAG augmentation.

    When dag_depth=0, behaves as a standard GPT model.
    When dag_depth>0, uses differentiable DAG for enhanced reasoning.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        # Validate dag_depth
        if config.dag_depth < 0:
            raise ValueError(
                f"DAG depth cannot be negative, got {config.dag_depth}. "
                f"Use dag_depth=0 for standard GPT or dag_depth≥1 for DAG-augmented GPT."
            )

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
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

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward_hidden(self, idx):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        return logits, loss

    def crop_block_size(self, block_size):
        """Adjust model to use a smaller block size if needed."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express as ratio of A100 peak FLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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
