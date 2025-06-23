from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, Block, GPTConfig


# ---------------------------------------------------------------------------
# ops
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
    return x / (y + eps)


def power(x, y, max_exp: float = 6.0):
    y = torch.clamp(y, -max_exp, max_exp)
    return torch.pow(torch.abs(x) + 1e-6, y)


def log(x, _unused=None, eps: float = 1e-8):
    return torch.log(torch.abs(x) + eps)


def max_op(x, y):
    return torch.max(x, y)


def min_op(x, y):
    return torch.min(x, y)


op_funcs = [add, identity, multiply, subtract, divide, power, log, max_op, min_op]


# ---------------------------------------------------------------------------
# controller: unchanged except it only sees EMBEDDINGS
# ---------------------------------------------------------------------------
class DAGController(nn.Module):
    def __init__(self, hidden_dim: int, n_ops: int):
        super().__init__()
        self.query_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.op_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.op_selector = nn.Linear(hidden_dim, n_ops)

        self.last_attn: torch.Tensor | None = None
        self.last_op_weights: torch.Tensor | None = None

    def forward(
        self, embeds: torch.Tensor, operand_ctx: torch.Tensor, op_ctx: torch.Tensor
    ):
        """embeds : (B , N , H)  – values are *not* needed here."""
        summary = embeds.mean(dim=1)  # (B , H)

        q1 = self.query_proj1(summary) + operand_ctx  # (B , H)
        q2 = self.query_proj2(summary) + operand_ctx  # (B , H)
        keys = self.key_proj(embeds)  # (B , N , H)

        att1 = torch.einsum("bnh,bh->bn", keys, q1) / (embeds.size(-1) ** 0.5)
        att2 = torch.einsum("bnh,bh->bn", keys, q2) / (embeds.size(-1) ** 0.5)

        attn1 = F.softmax(att1, dim=1)  # (B , N)
        attn2 = F.softmax(att2, dim=1)  # (B , N)

        # op selection
        op_q = self.op_query_proj(summary) + op_ctx
        att_op = torch.einsum("bnh,bh->bn", keys, op_q) / (embeds.size(-1) ** 0.5)
        attn_op = F.softmax(att_op, dim=1)  # (B , N)
        op_ctx_vec = torch.sum(attn_op.unsqueeze(-1) * embeds, dim=1)
        op_weights = F.softmax(self.op_selector(op_ctx_vec), dim=-1)  # (B , n_ops)

        self.last_attn = attn1.detach()
        self.last_op_weights = op_weights.detach()
        return attn1, attn2, op_weights  # return just the *distributions*


# ---------------------------------------------------------------------------
# Differentiable DAG with separate VALUE & EMBED streams
# ---------------------------------------------------------------------------
class DifferentiableDAG(nn.Module):
    def __init__(self, hidden_dim: int, num_steps: int, scalar_to_embed: nn.Module):
        super().__init__()
        self.num_steps = num_steps
        self.controller = DAGController(hidden_dim, len(op_funcs))
        self.step_emb = nn.Embedding(num_steps, hidden_dim)
        self.scalar_to_embed = scalar_to_embed

        self.embed_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim + len(op_funcs), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        embeds_list: list[torch.Tensor],  # each (B , H)
        values_list: list[torch.Tensor],  # each (B)  scalar
        operand_ctx: torch.Tensor,  # (B , H)
        op_ctx: torch.Tensor,  # (B , H)
        return_info: bool = False,
    ):
        attn_hist, op_hist = [], []
        B, _ = embeds_list[0].shape

        for step in range(self.num_steps):
            # stack existing nodes
            embeds = torch.stack(embeds_list, dim=1)  # (B , N , H)
            values = torch.stack(values_list, dim=1)  # (B , N)

            emb_step = self.step_emb.weight[step].unsqueeze(0).expand(B, -1)  # (B , H)

            # controller gives us weights over nodes
            att1, att2, op_weights = self.controller(
                embeds,
                operand_ctx + emb_step,
                op_ctx + emb_step,
            )

            # select embeddings
            e1 = torch.sum(att1.unsqueeze(-1) * embeds, dim=1)  # (B , H)
            e2 = torch.sum(att2.unsqueeze(-1) * embeds, dim=1)  # (B , H)

            # select values
            v1 = torch.sum(att1 * values, dim=1)  # (B)
            v2 = torch.sum(att2 * values, dim=1)  # (B)

            # compute new value
            outs = torch.stack([op(v1, v2) for op in op_funcs], dim=1)  # (B , n_ops)
            new_val = torch.sum(op_weights * outs, dim=1)  # (B)

            # Project to embedding space to inform the embedding combinator
            v1_embed = self.scalar_to_embed(v1.unsqueeze(-1))
            v2_embed = self.scalar_to_embed(v2.unsqueeze(-1))

            # compute new embed
            concat = torch.cat(
                [e1, e2, v1_embed, v2_embed, op_weights], dim=-1
            )  # (B , 2H + n_ops)
            delta = self.embed_mlp(concat)
            new_emb = 0.5 * (e1 + e2) + delta  # residual

            embeds_list.append(new_emb)
            values_list.append(new_val)

            if return_info:
                attn_hist.append(self.controller.last_attn)
                op_hist.append(self.controller.last_op_weights)

        if return_info:
            return embeds_list, values_list, attn_hist, op_hist
        return embeds_list, values_list


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
        attn_out, _ = self.attn(node_embeds, node_embeds, node_embeds)
        values = self.proj(attn_out).squeeze(-1)
        return values


# ---------------------------------------------------------------------------
# DAG-augmented GPT
# ---------------------------------------------------------------------------
@dataclass
class DAGGPTConfig(GPTConfig):
    dag_depth: int = 4


class DAGGPT(GPT):
    """GPT + differentiable DAG with separated values / embeddings."""

    def __init__(self, config: DAGGPTConfig):
        super().__init__(config)

        self.value_extractor = ValueExtractor(config)
        self.scalar_to_embed = nn.Linear(1, config.n_embd)

        self.dag = DifferentiableDAG(
            config.n_embd, config.dag_depth, self.scalar_to_embed
        )
        self.mix_gate = nn.Linear(config.n_embd * 2, 1)

        self.token_attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )
        self.value_attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )
        self.node_embed_block = Block(config)
        self.operand_ctx_block = Block(config)
        self.op_ctx_block = Block(config)
        self.post_dag_block = Block(config)

    # -------------------------------------------------------------

    def forward(self, idx, targets=None, return_dag_info: bool = False):
        _, T = idx.shape
        device = idx.device
        assert T <= self.config.block_size

        # ── GPT backbone ───────────────────────────────────────────
        pos = torch.arange(T, device=device)
        emb = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(emb)
        for blk in self.transformer.h:
            x = blk(x)
        hidden = self.transformer.ln_f(x)  # (B , T , H)

        # ── prepare node embeddings & scalar values ───────────────
        node_embeds = self.node_embed_block(
            hidden
        )  # Contains linguistic context specific to mathematics.
        node_values = self.value_extractor(
            node_embeds
        )  # (B, T) # Values derived from the mathematical linguistic context.

        embeds_list = [
            node_embeds[:, i, :] for i in range(T)
        ]  # length-T list of (B , H)
        values_list = [node_values[:, i] for i in range(T)]  # length-T list of (B)

        # guarantee at least two nodes
        while len(embeds_list) < 2:
            embeds_list.append(torch.zeros_like(embeds_list[0]))
            values_list.append(torch.zeros_like(values_list[0]))

        # Operand & operator chooser context
        operand_ctx = self.operand_ctx_block(node_embeds).mean(dim=1)  # (B , H)
        op_ctx = self.op_ctx_block(node_embeds).mean(dim=1)  # (B , H)

        # ── run DAG ───────────────────────────────────────────────
        dag_out = self.dag(
            embeds_list,
            values_list,
            operand_ctx,
            op_ctx,
            return_info=return_dag_info,
        )
        if return_dag_info:
            embeds_list, values_list, att_hist, op_hist = dag_out
        else:
            embeds_list, values_list = dag_out
            att_hist = op_hist = None

        # Project values into H-dim space
        value_tensor = torch.stack(values_list, dim=1).unsqueeze(-1)  # (B, N, 1)
        value_proj = self.scalar_to_embed(value_tensor)  # (B, N, H)

        # Aggregate (mean pool) the value embeddings into a single embedding
        attn_out, _ = self.value_attn(value_proj, value_proj, value_proj)
        pooled_values = attn_out.mean(dim=1)

        # Refine with post-dag block
        dag_sem = self.post_dag_block(pooled_values.unsqueeze(1)).squeeze(1)  # (B, H)

        # Fuse into hidden state
        gate = torch.sigmoid(
            self.mix_gate(torch.cat([hidden[:, -1, :], dag_sem], dim=-1))
        )
        fused = (1 - gate) * hidden[:, -1, :] + gate * dag_sem
        hidden = torch.cat([hidden[:, :-1, :], fused.unsqueeze(1)], dim=1)

        # ── keep a few salient activations for diagnostics ──────────────
        self.last_activations = {
            "node_embeds": node_embeds,
            "value_proj": value_proj,
            "operand_ctx": operand_ctx,
            "op_ctx": op_ctx,
        }

        # Prepare gradients dict
        self.dag_grads = {}

        # Register hooks to capture gradients after backward
        for name, t in self.last_activations.items():
            if t.requires_grad:

                def save_grad(grad, n=name):
                    self.dag_grads[n] = grad.detach().mean().item()

                t.register_hook(save_grad)

        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)

        if return_dag_info:
            return (
                logits,
                loss,
                {
                    "values": values_list,
                    "embeds": embeds_list,
                    "attn": att_hist,
                    "op": op_hist,
                },
            )
        return logits, loss

    # -----------------------------------------------------------------
    # extra_vals helper (unchanged – still works on stored activations)
    # -----------------------------------------------------------------
    def extra_vals(self) -> dict[str, float]:
        out = {}

        # Handle case where forward pass hasn't been called yet
        if not hasattr(self, "last_activations"):
            return out

        for name, act in self.last_activations.items():
            with torch.no_grad():
                prob = torch.softmax(act, dim=-1)
                ent = -(prob * (prob + 1e-8).log()).sum(-1).mean()
                out[f"dag_entropy/{name}"] = ent.item()
            if hasattr(self, "dag_grads") and name in self.dag_grads:
                out[f"dag_grad/{name}"] = self.dag_grads[name]
        return out
