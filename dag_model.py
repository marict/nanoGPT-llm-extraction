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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
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
# DAG operations
# ---------------------------------------------------------------------------
def add(x, y):
    return x + y


def identity(x, _):
    return x


def multiply(x, y):
    return x * y


def subtract(x, y):
    return x - y


def divide(x, y, eps: float = 1e-8):
    return torch.clamp(x / (y.abs() + eps), -1e6, 1e6)


def power(x, y, max_exp: float = 6.0):
    x_safe = torch.clamp(x.abs() + 1e-8, min=1e-8, max=1e3)
    y_safe = torch.clamp(y, min=-max_exp, max=max_exp)
    result = (x_safe**y_safe) * torch.sign(x)
    return torch.clamp(result, -1e6, 1e6)


def log(x, _unused=None, eps: float = 1e-8):
    return torch.clamp(torch.log(torch.clamp(torch.abs(x) + eps, min=eps)), -20.0, 20.0)


def max_op(x, y):
    return torch.maximum(x, y)


def min_op(x, y):
    return torch.minimum(x, y)


# op_funcs = [add, identity, multiply, subtract, divide, power, log, max_op, min_op]
# #op_names = [
#     "add",
#     "identity",
#     "multiply",
#     "subtract",
#     "divide",
#     "power",
#     "log",
#     "max_op",
#     "min_op",
# ]
# Minimal set of ops for testing
op_funcs = [add, identity, multiply, subtract]
op_names = ["add", "identity", "multiply", "subtract"]


def safe_clamp(logits: torch.Tensor) -> torch.Tensor:
    # choose a head-room margin so exp(max_val) is still finite
    if logits.dtype == torch.float16:
        max_val = 8.0  # exp(8)  ≈ 2981  fits in fp16
    elif logits.dtype == torch.bfloat16:
        max_val = 20.0  # exp(20) ≈ 4.85e8 fits in bf16
    else:  # float32 or higher
        max_val = 40.0  # exp(40) ≈ 2.35e17 fits in fp32
    return logits.clamp(-max_val, max_val)


# ---------------------------------------------------------------------------
# controller: unchanged except it only sees EMBEDDINGS
# ---------------------------------------------------------------------------
class DAGController(nn.Module):
    def __init__(self, hidden_dim: int, n_ops: int, temperature: float = 2.0):
        super().__init__()
        # Use MultiheadAttention for operand selection
        self.operand_attn1 = nn.MultiheadAttention(
            hidden_dim, num_heads=1, batch_first=True
        )
        self.operand_attn2 = nn.MultiheadAttention(
            hidden_dim, num_heads=1, batch_first=True
        )

        # Operation selection
        self.op_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.op_selector = nn.Linear(hidden_dim, n_ops)
        self.temperature = temperature

        self.last_attn: torch.Tensor | None = None
        self.last_op_weights: torch.Tensor | None = None
        self.last_op_logits: torch.Tensor | None = None
        self.last_op_logits_with_grad: torch.Tensor | None = None

    def forward(
        self,
        embeds: torch.Tensor,
        operand_ctx: torch.Tensor,
        op_ctx: torch.Tensor,
    ):
        # embeds: (B, N, H)
        # operand_ctx: (B, H)
        # op_ctx: (B, H)
        B, N, H = embeds.shape

        # Use MultiheadAttention for operand selection
        # Query is the context, key/value are the embeddings
        query1 = operand_ctx.unsqueeze(1)  # (B, 1, H)
        query2 = operand_ctx.unsqueeze(1)  # (B, 1, H)

        # Get attention outputs and weights
        _, att_weights1 = self.operand_attn1(query1, embeds, embeds)  # (B, 1, N)
        _, att_weights2 = self.operand_attn2(query2, embeds, embeds)  # (B, 1, N)

        # Squeeze to get (B, N) and apply Gumbel softmax
        att_weights1 = att_weights1.squeeze(1)  # (B, N)
        att_weights2 = att_weights2.squeeze(1)  # (B, N)

        # Apply safe clamping and Gumbel softmax
        att_weights1 = safe_clamp(att_weights1)
        att_weights2 = safe_clamp(att_weights2)

        # Use deterministic Gumbel softmax for reproducible results
        # Save current RNG state
        rng_state = torch.get_rng_state()

        # Set deterministic seed based on input for reproducibility
        seed = int(torch.sum(att_weights1 * 1000).item()) % 2**31
        torch.manual_seed(seed)
        att1 = F.gumbel_softmax(att_weights1, tau=self.temperature, hard=False)

        seed = int(torch.sum(att_weights2 * 1000).item()) % 2**31
        torch.manual_seed(seed)
        att2 = F.gumbel_softmax(att_weights2, tau=self.temperature, hard=False)

        # Restore RNG state
        torch.set_rng_state(rng_state)

        # Prevent gradient issues by ensuring selections are valid
        att1 = att1 + 1e-10 * torch.ones_like(att1)
        att2 = att2 + 1e-10 * torch.ones_like(att2)

        # Operation selection
        op_query = self.op_query_proj(op_ctx)  # (B, H)
        op_logits = self.op_selector(op_query)  # (B, n_ops)
        op_logits = safe_clamp(op_logits)

        # Use deterministic Gumbel-Softmax for discrete operation selection
        rng_state_op = torch.get_rng_state()
        seed_op = int(torch.sum(op_logits * 1000).item()) % 2**31
        torch.manual_seed(seed_op)
        op_weights = F.gumbel_softmax(op_logits, tau=self.temperature, hard=True)
        torch.set_rng_state(rng_state_op)

        # Prevent gradient issues by ensuring op weights are valid
        op_weights = op_weights + 1e-10 * torch.ones_like(op_weights)

        # Store for logging
        self.last_attn = torch.stack([att1, att2], dim=1)  # (B, 2, N)
        self.last_op_weights = op_weights
        self.last_op_logits = op_logits
        # Store op_logits with gradients for gradient tracking
        self.last_op_logits_with_grad = op_logits

        return att1, att2, op_weights


# ---------------------------------------------------------------------------
# DAG computation
# ---------------------------------------------------------------------------
class DifferentiableDAG(nn.Module):
    def __init__(
        self,
        config,
        scalar_to_embed: nn.Module,
    ):
        super().__init__()
        self.hidden_dim = config.n_embd
        self.dag_depth = config.dag_depth
        self.scalar_to_embed = scalar_to_embed
        self.temperature = config.gumbel_temperature

        # Initialize controller
        self.controller = DAGController(
            self.hidden_dim, len(op_funcs), self.temperature
        )

        # Step embeddings for each step in the DAG
        self.step_emb = nn.Embedding(self.dag_depth, self.hidden_dim)

        # MLP to combine all inputs into final node embedding
        input_dim = 4 * self.hidden_dim + len(
            op_funcs
        )  # 4 embeds (e1, e2, v1_emb, v2_emb) + op logits
        self.embed_mlp = nn.Linear(input_dim, self.hidden_dim)

        # Components moved from GPT class
        self.node_embed_block = Block(config)
        self.post_dag_block = Block(config)
        self.dag_norm = LayerNorm(config.n_embd, bias=config.bias)

        # Value extractor and context selectors
        self.value_extractor = ValueExtractor(config)
        self.operand_ctx_selector = MaskedContextSelector(config, self.temperature)
        self.op_ctx_selector = MaskedContextSelector(config, self.temperature)

        # Initialize node storage
        self.node_embeds = None  # Will be set in forward pass
        self.node_values = None  # Will be set in forward pass

    def forward(
        self,
        original_hidden: torch.Tensor,  # (B, T, H) - batch, time, hidden
    ):
        B, T, H = original_hidden.shape
        device = original_hidden.device

        # Process original hidden states to get node embeddings and values
        node_embeds = self.node_embed_block(original_hidden)  # (B, T, H)

        # Initialize node storage tensors
        # Start with initial nodes for each token position
        # Shape: (B, N, T, H) for embeddings, (B, N, T) for values
        # N = dag_depth + 1 nodes
        N = self.dag_depth + 1
        initial_embeds = node_embeds.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T, H)
        initial_values = (
            self.value_extractor(node_embeds).unsqueeze(1).expand(-1, N, -1)
        )  # (B, N, T)

        # Get contexts for operand and operation selection using masked attention
        # For each position t, context comes from positions 0 to t-1
        causal_lengths = [t for t in range(T)]  # Position t can see 0 to t-1 tokens

        # Use masked context selectors with Gumbel softmax
        operand_ctx = self.operand_ctx_selector(
            original_hidden, causal_lengths
        )  # (B, T, H)
        op_ctx = self.op_ctx_selector(original_hidden, causal_lengths)  # (B, T, H)

        # Store the current node state
        self.node_embeds = initial_embeds
        self.node_values = initial_values

        # Start with current nodes
        current_embeds = initial_embeds
        current_values = initial_values

        # Run dag_depth iterations to add dag_depth new nodes
        for step in range(self.dag_depth):
            # Initialize output tensors with one more node than current
            new_embeds = torch.zeros(
                B, current_embeds.shape[1] + 1, T, H, device=device
            )
            new_values = torch.zeros(B, current_values.shape[1] + 1, T, device=device)

            # Copy existing nodes
            new_embeds[:, :-1] = current_embeds
            new_values[:, :-1] = current_values

            for t in range(T):
                # Create causal mask for this token position
                # For each token t, we want access to ALL nodes from previous tokens
                if t == 0:
                    # For the first token, use current existing nodes (for single token case)
                    # This allows T=1 sequences to still do DAG operations
                    flat_embeds = current_embeds[:, :, t].reshape(B, -1, H)  # (B, N, H)
                    flat_values = current_values[:, :, t].reshape(B, -1)  # (B, N)
                else:
                    # For t>0, we can use all nodes from previous tokens (strictly causal)
                    flat_embeds = current_embeds[:, :, :t].reshape(
                        B, -1, H
                    )  # (B, N*t, H)
                    flat_values = current_values[:, :, :t].reshape(B, -1)  # (B, N*t)

                # Skip if no nodes available
                if flat_embeds.numel() == 0:
                    new_embeds[:, -1, t] = 0
                    new_values[:, -1, t] = 0
                    continue

                # Get controller decisions
                att1, att2, op_w = self.controller(
                    flat_embeds,
                    operand_ctx[:, t]
                    + self.step_emb.weight[step].unsqueeze(0),  # (B, H)
                    op_ctx[:, t] + self.step_emb.weight[step].unsqueeze(0),  # (B, H)
                )

                # Select embeddings and values from previous nodes
                e1 = (att1.unsqueeze(-1) * flat_embeds).sum(1)  # (B, H)
                e2 = (att2.unsqueeze(-1) * flat_embeds).sum(1)  # (B, H)
                v1 = (att1 * flat_values).sum(1)  # (B)
                v2 = (att2 * flat_values).sum(1)  # (B)

                # Compute operation outputs
                outs = torch.stack([op(v1, v2) for op in op_funcs], dim=1)  # (B, n_ops)
                new_val = (op_w * outs).sum(1)  # (B)

                # Project scalar values to embedding space
                v1_emb = self.scalar_to_embed(v1.unsqueeze(-1))  # (B, H)
                v2_emb = self.scalar_to_embed(v2.unsqueeze(-1))  # (B, H)

                # Compute new embedding
                concat = torch.cat([e1, e2, v1_emb, v2_emb, op_w], dim=-1)
                delta = self.embed_mlp(concat)
                new_emb = 0.5 * (e1 + e2) + delta  # residual

                # Store new node
                new_embeds[:, -1, t] = new_emb
                new_values[:, -1, t] = new_val

            # Update current nodes for next step
            current_embeds = new_embeds
            current_values = new_values

        # Process each token position to create final hidden states
        new_hidden_list = []
        for t in range(T):
            # Get the final node embedding for this position
            final_node_embed = current_embeds[:, -1, t]  # (B, H)

            # Process final node embedding through post-DAG block and normalization
            dag_sem = self.post_dag_block(final_node_embed.unsqueeze(1)).squeeze(1)
            dag_sem = self.dag_norm(dag_sem)
            new_hidden_list.append(dag_sem)

        # Reconstruct hidden tensor from list
        final_hidden = torch.stack(new_hidden_list, dim=1)  # (B, T, H)

        return final_hidden, current_embeds, current_values


# ---------------------------------------------------------------------------
# Learned transformer -> MLP process for converting initial embeddings into values.
# ---------------------------------------------------------------------------
class ValueExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )
        self.proj = nn.Linear(config.n_embd, 1)

    def forward(self, node_embeds):
        # node_embeds: (B, T, H)
        B, T, H = node_embeds.shape
        device = node_embeds.device

        # Create causal attention mask so position N can only attend to positions 0 through N-1
        # For PyTorch MultiheadAttention, mask shape is (T, T) where True means "ignore"
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        attn_out, _ = self.attn(
            node_embeds, node_embeds, node_embeds, attn_mask=causal_mask
        )
        values = self.proj(attn_out).squeeze(-1)
        return values


# ---------------------------------------------------------------------------
# Context selectors using masked MultiheadAttention + Gumbel softmax
# ---------------------------------------------------------------------------
class MaskedContextSelector(nn.Module):
    """Replaces Block-based context selection with masked attention + Gumbel softmax."""

    def __init__(self, config, temperature: float = 2.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )
        self.temperature = temperature

    def forward(self, hidden_states, causal_lengths):
        # hidden_states: (B, T, H) - all token representations
        # causal_lengths: list of length T, where causal_lengths[t] = number of tokens available at position t
        B, T, H = hidden_states.shape
        device = hidden_states.device

        # Create causal mask for each position
        # We'll process each position separately to handle variable causal lengths
        outputs = []

        for t in range(T):
            if causal_lengths[t] == 0:
                # No previous context available, use zero
                outputs.append(torch.zeros(B, H, device=device))
            else:
                # Use causal context up to position t
                context_len = causal_lengths[t]
                context = hidden_states[:, :context_len, :]  # (B, context_len, H)
                query = hidden_states[
                    :, [t], :
                ]  # (B, 1, H) - current position as query

                # Apply attention
                attn_out, attn_weights = self.attn(query, context, context)  # (B, 1, H)

                # Apply Gumbel softmax to attention weights for discrete selection
                if attn_weights.dim() == 3:  # (B, 1, context_len)
                    attn_weights = attn_weights.squeeze(1)  # (B, context_len)

                # Safe clamp before Gumbel softmax
                attn_weights = safe_clamp(attn_weights)

                # Use deterministic Gumbel softmax
                rng_state = torch.get_rng_state()
                seed = int(torch.sum(attn_weights * 1000).item()) % 2**31
                torch.manual_seed(seed)
                discrete_weights = F.gumbel_softmax(
                    attn_weights, tau=self.temperature, hard=False
                )
                torch.set_rng_state(rng_state)

                # Apply discrete weights to get final context
                final_context = (discrete_weights.unsqueeze(-1) * context).sum(
                    dim=1
                )  # (B, H)
                outputs.append(final_context)

        return torch.stack(outputs, dim=1)  # (B, T, H)


# ---------------------------------------------------------------------------
# DAG Mixer for combining original and DAG-processed hidden states
# ---------------------------------------------------------------------------
class DAGMixer(nn.Module):
    """Handles mixing of original transformer hidden states with DAG-processed states."""

    def __init__(self, config):
        super().__init__()
        self.mix_gate = nn.Linear(config.n_embd * 2, 1)

    def forward(self, original_hidden, dag_hidden):
        """
        Mix original and DAG-processed hidden states.

        Args:
            original_hidden: (B, T, H) - Original transformer hidden states
            dag_hidden: (B, T, H) - DAG-processed hidden states

        Returns:
            mixed_hidden: (B, T, H) - Mixed hidden states
            gate_values: list of gate values for logging
        """
        B, T, H = original_hidden.shape

        # Process each token position to mix original and DAG hidden states
        new_hidden_list = []
        gate_values = []

        for t in range(T):
            # Mix original and DAG-processed hidden states
            gate = torch.sigmoid(
                self.mix_gate(
                    torch.cat([original_hidden[:, t], dag_hidden[:, t]], dim=-1)
                )
            )
            fused_hidden = (1 - gate) * original_hidden[:, t] + gate * dag_hidden[:, t]
            new_hidden_list.append(fused_hidden)
            gate_values.append(gate.detach())

        # Reconstruct hidden tensor from list
        mixed_hidden = torch.stack(new_hidden_list, dim=1)

        return mixed_hidden, gate_values


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
    gumbel_temperature: float = 2.0


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
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # DAG-specific components (only if dag_depth > 0)
        if config.dag_depth > 0:
            self.scalar_to_embed = nn.Linear(1, config.n_embd)
            self.dag = DifferentiableDAG(config, self.scalar_to_embed)
            self.dag_mixer = DAGMixer(config)

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

    def get_node_values_list(self) -> list[float]:
        """
        Get the node values from the last forward pass as a list of floats.
        Used for logging during evaluation.

        Returns:
            List of node values as floats, empty list if no forward pass or dag_depth=0.
            For causal DAG, returns the final node value for each token position.
        """
        if self.config.dag_depth == 0:
            return []

        if not hasattr(self, "last_values_list") or self.last_values_list is None:
            return []

        # For causal DAG, last_values_list is a tensor of shape (B, T)
        # where B is batch size and T is sequence length
        # We want to return the mean across batch for each token position
        if isinstance(self.last_values_list, torch.Tensor):
            return [val.mean().item() for val in self.last_values_list.transpose(0, 1)]

        # Fallback for non-tensor case (shouldn't happen with causal DAG)
        result = []
        for val in self.last_values_list:
            if hasattr(val, "item"):
                if val.numel() == 1:
                    result.append(val.item())
                else:
                    result.append(val.mean().item())
            else:
                result.append(float(val))
        return result

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

        # Run DAG processing (handles everything internally)
        dag_hidden, final_embeds, final_values = self.dag(original_hidden)

        # Store for logging
        self.last_values_list = final_values[
            :, -1
        ]  # Take last node's values for each position

        # Mix original and DAG hidden states using the DAGMixer
        hidden, gate_values = self.dag_mixer(original_hidden, dag_hidden)

        # Store gate and norm values for logging (use last token's values)
        self.last_gate_values = gate_values[-1] if gate_values else None
        self.last_norm_values = {
            "hidden": original_hidden[:, -1].norm(dim=-1).mean().detach(),
            "dag_sem": dag_hidden[:, -1].norm(dim=-1).mean().detach(),
            "fused": hidden[:, -1].norm(dim=-1).mean().detach(),
        }

        logits = self.lm_head(hidden)
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

            # apply softmax to convert logits to (normalized) probabilities
            logits = torch.clamp(logits, -50.0, 50.0)
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
