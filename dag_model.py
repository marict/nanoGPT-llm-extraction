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


# ---------------------------------------------------------------------------
# DAG Log-space arithmetic utilities
# ---------------------------------------------------------------------------
LOG_LIM = 15.0  # Bound on log-magnitudes to avoid numerical instabilities
SIGN_SCALE = 5.0  # Controls softness of sign prediction at intake
GRAD_CAP = 1.0  # try 0.1–10.0 depending on bfloat16/float16 range


def _clip_log(log_t: torch.Tensor) -> torch.Tensor:
    """Symmetric tanh clipping for log magnitudes."""
    return torch.tanh(log_t / LOG_LIM) * LOG_LIM


# ---------------------------------------------------------------------------
# Smooth log rescaling (replaces hard _clip_log inside DAG loop)
# ---------------------------------------------------------------------------


def _rms_rescale(log_stack: list[torch.Tensor]) -> None:
    """In-place rescale of the *last* entry in log_stack so that the running RMS
    over the stack equals LOG_LIM. Works per-token, keeps gradients."""

    # Compute rms over current stack (B,T)
    logs = torch.stack(log_stack, dim=-1)  # (B,T,S)
    rms = torch.sqrt((logs**2).mean(dim=-1, keepdim=True) + 1e-6)
    scale = (LOG_LIM / rms).clamp(max=1.0)  # only scale down
    # Apply scaling only to the most recent entry to avoid in-place ops on saved tensors
    if (scale < 1).any():  # scale is (B,T,1)
        log_stack[-1] = log_stack[-1] * scale.squeeze(-1)


# ---------------------------------------------------------------------------
# Debug utility
# ---------------------------------------------------------------------------


def _debug_check(op_name: str, *tensors: torch.Tensor):
    """Verbose debug checker.
    Prints detailed information (shape, dtype, min/mean/max, and a handful of offending
    indices/values) the moment a non-finite value is detected in *any* of the provided
    tensors. This makes it easier to pinpoint the precise source of NaNs/Infs during
    training without flooding the log with entire tensor dumps.
    """
    for idx, t in enumerate(tensors):
        # Skip None entries just in case
        if t is None:
            continue

        # Fast path: only continue if something is non-finite
        if torch.isfinite(t).all():
            continue

        nan_mask = torch.isnan(t)
        inf_mask = torch.isinf(t)
        nan_cnt = int(nan_mask.sum())
        inf_cnt = int(inf_mask.sum())

        # Basic tensor meta
        print(
            f"[DEBUG] {op_name}: tensor#{idx} shape={tuple(t.shape)} dtype={t.dtype} "
            f"-> NaN: {nan_cnt}  Inf: {inf_cnt}"
        )

        # Compute simple stats on finite values (cast to fp32 for safety)
        finite_vals = t[torch.isfinite(t)].float()
        if finite_vals.numel() > 0:
            stats = (
                float(finite_vals.min()),
                float(finite_vals.mean()),
                float(finite_vals.max()),
            )
            print(f"           finite stats (min/mean/max): {stats}")

        # Show up to first 5 bad indices & their values for pinpointing
        bad_idx = (~torch.isfinite(t)).nonzero(as_tuple=False)
        for j, coord in enumerate(bad_idx[:5]):
            coord_tuple = tuple(coord.tolist())
            bad_val = t[coord_tuple].item()
            print(f"           [{j}] index={coord_tuple} value={bad_val}")

        raise RuntimeError("NaN/Inf detected; see logs above for details")


def multiply_log_space(
    sx: torch.Tensor, lx: torch.Tensor, sy: torch.Tensor, ly: torch.Tensor
):
    sgn_out = sx * sy
    log_out = lx + ly
    log_out = _clip_log(log_out)
    _debug_check("multiply_log_space", sgn_out, log_out)
    return sgn_out, log_out


def divide_log_space(
    sx: torch.Tensor, lx: torch.Tensor, sy: torch.Tensor, ly: torch.Tensor
):
    sgn_out = sx * sy
    log_out = lx - ly
    log_out = _clip_log(log_out)
    _debug_check("divide_log_space", sgn_out, log_out)
    return sgn_out, log_out


# Helper when operands share the same sign


def _add_logs_same_sign(sgn: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor):
    # Perform high-precision accumulation to avoid bf16 overflows
    lx32, ly32 = lx.float(), ly.float()
    l_out = torch.logaddexp(lx32, ly32).to(lx.dtype)
    return sgn, _clip_log(l_out)


def add_log_space(
    sx: torch.Tensor, lx: torch.Tensor, sy: torch.Tensor, ly: torch.Tensor
):
    """Addition in log-space with sign handling (vectorised).
    Args:
        sx: (B,T) - signed log magnitudes of first operand
        lx: (B,T) - log magnitudes of first operand
        sy: (B,T) - signed log magnitudes of second operand
        ly: (B,T) - log magnitudes of second operand

    Returns:
        s_out: (B,T) - signed log magnitudes of result
        l_out: (B,T) - log magnitudes of result
    """
    zero_x = sx == 0
    zero_y = sy == 0

    # Initialise outputs
    s_out = torch.zeros_like(sx)
    l_out = torch.zeros_like(lx)

    # Only-x / only-y cases
    only_x = (~zero_x) & zero_y
    only_y = zero_x & (~zero_y)
    s_out = torch.where(only_x, sx, s_out)
    l_out = torch.where(only_x, lx, l_out)
    s_out = torch.where(only_y, sy, s_out)
    l_out = torch.where(only_y, ly, l_out)

    # Both non-zero
    both_nz = (~zero_x) & (~zero_y)
    same_sign = (sx * sy) > 0
    same_branch = both_nz & same_sign
    if same_branch.any():
        # Use a single shared sign to guard against rare numerical mismatch
        common_sgn = torch.sign(sx)  # sx and sy have same sign in this branch
        s_same, l_same = _add_logs_same_sign(common_sgn, lx, ly)
        s_out = torch.where(same_branch, s_same, s_out)
        l_out = torch.where(same_branch, l_same, l_out)

    # Opposite sign branch
    opp_branch = both_nz & (~same_sign)
    if opp_branch.any():
        bigger_is_x = lx >= ly
        big_log = torch.where(bigger_is_x, lx, ly)
        small_log = torch.where(bigger_is_x, ly, lx)
        big_sgn = torch.where(bigger_is_x, sx, sy)

        # Ensure delta is sufficiently negative so that exp(delta) < 1 but also
        # not *too* close to zero, where the derivative of log1p(-exp(delta))
        # explodes and causes huge gradients. A small safe margin (-1e-3)
        # prevents this while introducing only <0.1% relative error.
        delta32 = (small_log - big_log).float().clamp(max=-1e-3, min=-LOG_LIM)
        diff32 = torch.log1p(-torch.exp(delta32))  # safe in fp32
        diff = diff32.to(lx.dtype)
        # Perfect cancellation when operands equal magnitude but opposite sign
        zero_res = small_log == big_log
        new_log = big_log + diff
        new_sgn = big_sgn
        new_sgn = torch.where(zero_res, torch.zeros_like(new_sgn), new_sgn)
        new_log = torch.where(zero_res, torch.zeros_like(new_log), new_log)

        s_out = torch.where(opp_branch, new_sgn, s_out)
        l_out = torch.where(opp_branch, new_log, l_out)

    l_out = _clip_log(l_out)
    _debug_check("add_log_space", s_out, l_out)
    return s_out, l_out


def subtract_log_space(
    sx: torch.Tensor, lx: torch.Tensor, sy: torch.Tensor, ly: torch.Tensor
):
    # Subtraction = addition with negated second operand
    s_out, l_out = add_log_space(sx, lx, -sy, ly)
    _debug_check("subtract_log_space", s_out, l_out)
    return s_out, l_out


def identity_log_space(sx: torch.Tensor, lx: torch.Tensor, *_):
    return sx, lx


# Replace op list with log-space versions (order important for op_names)
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
    """Predicts complete DAG execution plans for all tokens in batch."""

    def __init__(self, config, temperature: float = 10):
        super().__init__()
        self.dag_depth = config.dag_depth
        self.temperature = temperature
        self.block_size = config.block_size

        # Token-local scratch length: initial value + one new value per step
        self.num_scratch_nodes = config.dag_depth + 1
        self.n_ops = len(op_funcs)  # Number of operations

        # Total plan dimensions per token per step: [operand1, operand2, operation]
        self.plan_dim = 2 * self.num_scratch_nodes + self.n_ops

        # Predictor: hidden_state (+ sign, log) -> full DAG plan
        self.predictor = nn.Sequential(
            nn.Linear(config.n_embd + 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.dag_depth * self.plan_dim),
        )

        # Store last predictions for logging
        self.last_operation_probs_full: torch.Tensor | None = None
        self.last_operand1_probs: torch.Tensor | None = None
        self.last_operand2_probs: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, seed_feat: torch.Tensor):
        """
        Predict complete DAG plans for all tokens, maintaining causality.

        Args:
            hidden_states: (B, T, H) - hidden states for all tokens
            seed_feat: (B, T, 2) - [sign, log] features per token (log already clipped)
        Returns:
            operand1_probs: (B, T, dag_depth, max_nodes) - probabilities for operand 1
            operand2_probs: (B, T, dag_depth, max_nodes) - probabilities for operand 2
            operation_probs: (B, T, dag_depth, n_ops) - probabilities for operations
        """
        # Vectorised forward pass (no explicit Python loop over tokens)
        B, T, H = hidden_states.shape

        # --- concatenate sign & log with hidden state -----------------------
        mixed = torch.cat([hidden_states, seed_feat], dim=-1)  # (B,T,H+2)

        # Since B x T is processed independantly, this doesn't leak causality.
        mixed = mixed.reshape(B * T, H + 2)
        raw_plan = self.predictor(mixed)
        raw_plan = raw_plan.view(
            B, T, self.dag_depth, self.plan_dim
        )  # (B, T, D, plan_dim)
        dag_plan_logits = raw_plan  # alias for clarity

        # Split into operand and operation logits
        operand1_logits = dag_plan_logits[..., : self.num_scratch_nodes]
        operand2_logits = dag_plan_logits[
            ..., self.num_scratch_nodes : 2 * self.num_scratch_nodes
        ]
        operation_logits = dag_plan_logits[
            ...,
            2 * self.num_scratch_nodes : 2 * self.num_scratch_nodes + self.n_ops,
        ]

        # Clamp logits before softmax for numerical stability
        operand1_logits = safe_clamp(operand1_logits)
        operand2_logits = safe_clamp(operand2_logits)
        operation_logits = safe_clamp(operation_logits)

        # Standard softmax with temperature scaling
        # Scale to prevent underflow.
        scale = math.sqrt(self.n_ops)  # or math.log(self.n_ops)
        operand1_probs = F.softmax(operand1_logits * scale / self.temperature, dim=-1)
        operand2_probs = F.softmax(operand2_logits * scale / self.temperature, dim=-1)
        operation_probs = F.softmax(operation_logits * scale / self.temperature, dim=-1)

        # Cache tensors for external logging/debugging
        self.last_operation_probs_full = operation_probs
        self.last_operand1_probs = operand1_probs
        self.last_operand2_probs = operand2_probs

        return operand1_probs, operand2_probs, operation_probs


class ScalarToEmbed(nn.Module):
    """Map sign & log magnitude → H-dim embedding using Fourier bases.

    Expects input tensor of shape (B,T,2) where:
       feat[...,0] = sign ∈ [-1,1]
       feat[...,1] = log magnitude ≥ 0 (already clipped)
    """

    def __init__(self, hidden: int, sigma=1.0, feat_dim: int = 32):
        super().__init__()
        self.ff = PositionalEncoding(sigma, feat_dim)
        # +2 for sign and raw log magnitude, +2*feat_dim for periodic encodings
        self.proj = nn.Linear(2 * feat_dim + 2, hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # (B,T,2)
        sign = feat[..., 0:1]  # preserve dims
        logmag = feat[..., 1:2]
        trig = self.ff(logmag)  # (B,T,2*feat_dim)
        concat = torch.cat((sign, logmag, trig), dim=-1)  # (B,T,2*feat_dim+2)
        return self.proj(concat)  # (B,T,H)


# ---------------------------------------------------------------------------
# DAG computation
# ---------------------------------------------------------------------------
class DifferentiableDAG(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.hidden_dim = config.n_embd
        self.dag_depth = config.dag_depth
        self.num_scratch_nodes = config.dag_depth + 1
        self.temperature = config.softmax_temperature

        self.pre_dag = Block(config)
        self.post_dag = Block(config)

        # Initialize plan predictor
        self.plan_predictor = DAGPlanPredictor(config, self.temperature)

        self.scalar_to_embed = ScalarToEmbed(config.n_embd)
        # Seed head: LayerNorm → 2-dim projection
        self.seed_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.embed_to_signlog = nn.Linear(config.n_embd, 2)

    def forward(
        self,
        original_hidden: torch.Tensor,  # (B, T, H) - batch, time, hidden
    ):
        # Pre-process hidden states before value extraction
        incoming_hidden = self.pre_dag(original_hidden)

        _debug_check("incoming_hidden", incoming_hidden)

        # Predict sign & magnitude in bounded range
        seed_h = self.seed_norm(incoming_hidden)

        _debug_check("seed_h", seed_h)

        signlog = self.embed_to_signlog(seed_h)  # (B,T,2)

        _debug_check("signlog", signlog)
        sign_logits = signlog[..., 0]
        mag_logits = signlog[..., 1]

        # Sign in (−1,1) with optional sharpness scaling
        init_sgn = torch.tanh(sign_logits * SIGN_SCALE)

        # Magnitude: map (−∞,∞)→(0,LOG_LIM) via sigmoid
        init_log = LOG_LIM * torch.sigmoid(mag_logits)

        _debug_check("seed", init_sgn, init_log)

        # Signed log magnitude seed for plan predictor
        seed_feat = torch.stack((init_sgn, init_log), dim=-1)  # (B,T,2)

        # Scratch stacks (append-only)
        sgn_stack = [init_sgn]
        log_stack = [init_log]

        # Generate DAG plan (vectorised) using signed-log seed
        operand1_probs, operand2_probs, operation_probs = self.plan_predictor(
            incoming_hidden, seed_feat
        )

        for step in range(self.dag_depth):
            att1 = operand1_probs[:, :, step, :]  # (B,T,S)
            att2 = operand2_probs[:, :, step, :]
            op_w = operation_probs[:, :, step, :]  # (B,T,n_ops)

            S_curr = len(sgn_stack)
            att1_step = att1[..., :S_curr]
            att2_step = att2[..., :S_curr]

            current_sgn = torch.stack(sgn_stack, dim=-1)  # (B,T,S)
            current_log = torch.stack(log_stack, dim=-1)  # (B,T,S)

            # Soft-attention may blend sign information; discretise with sign() to keep {-1,0,1}
            v1_sgn_raw = (att1_step * current_sgn).sum(-1)
            v2_sgn_raw = (att2_step * current_sgn).sum(-1)
            v1_sgn = torch.sign(v1_sgn_raw)
            v1_sgn = torch.where(v1_sgn == 0, torch.ones_like(v1_sgn), v1_sgn)
            v2_sgn = torch.sign(v2_sgn_raw)
            v2_sgn = torch.where(v2_sgn == 0, torch.ones_like(v2_sgn), v2_sgn)

            # Log magnitudes are mixed linearly; this is a heuristic but preserves differentiability
            v1_log = (att1_step * current_log).sum(-1)
            v2_log = (att2_step * current_log).sum(-1)

            # Apply all operations in parallel in log-space
            outs = [op(v1_sgn, v1_log, v2_sgn, v2_log) for op in op_funcs]
            outs_sgn, outs_log = zip(*outs)  # each element (B,T)
            outs_sgn_t = torch.stack(list(outs_sgn), dim=-1)  # (B,T,n_ops)
            outs_log_t = torch.stack(list(outs_log), dim=-1)  # (B,T,n_ops)

            new_sgn = (op_w * outs_sgn_t).sum(-1)
            new_log = (op_w * outs_log_t).sum(-1)

            sgn_stack.append(new_sgn)
            log_stack.append(new_log)

            # Smoothly rescale logs to keep RMS within bounds (in-place)
            _rms_rescale(log_stack)

        final_sgn = sgn_stack[-1]
        final_log = log_stack[-1]

        final_feat = torch.stack((final_sgn, final_log), dim=-1)  # (B,T,2)

        # Convert final nodes to hidden states.
        final_hidden = self.scalar_to_embed(final_feat)  # (B,T,H)

        # Mix token information via post dag block.
        final_hidden = self.post_dag(final_hidden)

        # Cache for logging
        self.final_hidden = final_hidden
        # Store signed-log for logging compatibility
        z_stack = [sgn_stack[i] * log_stack[i] for i in range(len(sgn_stack))]
        self.final_values = torch.stack(z_stack, dim=1)  # (B,S,T)

        return final_hidden


# ---------------------------------------------------------------------------
# GPT configuration
# ---------------------------------------------------------------------------
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
    dag_depth: int = 4  # 0 = standard GPT, >0 = DAG-augmented GPT
    softmax_temperature: float = 20.0


# ---------------------------------------------------------------------------
# Main GPT model (with optional DAG augmentation)
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    """GPT Language Model with optional DAG augmentation.

    When dag_depth=0, behaves as a standard GPT model.
    When dag_depth>0, uses differentiable DAG for enhanced reasoning.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
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

        # DAG augmentation mode - simplified interface
        original_hidden = hidden.clone()  # Keep original for mixing

        # Store original hidden states for logging
        self.last_original_hidden = original_hidden

        _debug_check("original_hidden", original_hidden)

        # Run DAG processing (handles everything internally)
        dag_hidden = self.dag(original_hidden)

        # Clamp only the gradient flowing *out* of the DAG (helps stabilise training)
        dag_hidden.register_hook(lambda g: g.clamp_(min=-GRAD_CAP, max=GRAD_CAP))

        # Store complete tensor for detailed analysis (before mixing)
        self.final_values = self.dag.final_values

        # ------------------------------------------------------------------ #
        # Per-token gating mechanism
        # gate_{B,T,1} = σ( mean_dag + mean_orig ) where
        #   mean_dag  = ⟨dag_hidden * w_d⟩_H
        #   mean_orig = ⟨original_hidden * w_o⟩_H
        # Both w_d and w_o are learnable vectors of size H.
        # ------------------------------------------------------------------ #
        gate_logits = (
            (dag_hidden * self.gate_w_d).mean(-1, keepdim=True)
            + (original_hidden * self.gate_w_o).mean(-1, keepdim=True)
            + self.gate_b
        )
        gate = torch.sigmoid(gate_logits)  # (B,T,1) in (0,1)

        # Store gate for logging
        self.last_gate = gate

        mixed_hidden = original_hidden + gate * dag_hidden

        # Store mixed hidden states for logging
        self.last_mixed_hidden = mixed_hidden

        logits = self.lm_head(mixed_hidden)
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
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
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # Add gate parameters to decay_params
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Normalize logits magnitude instead of hard clipping
            logits_std = logits.std(dim=-1, keepdim=True).clamp_min(1e-3)
            logits = logits / logits_std
            probs = F.softmax(logits, dim=-1)

            # Handle edge cases (NaN/inf values)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            else:
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
