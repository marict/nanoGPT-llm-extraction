from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Differentiable DAG components
# -----------------------------------------------------------------------------


def add(x, y):
    return x + y


def multiply(x, y):
    return x * y


def subtract(x, y):
    return x - y


def divide(x, y, eps: float = 1e-8):
    return x / (y + eps)


def identity(x, _unused=None):
    return x


def power(x, y, max_exp: float = 6.0):
    """Element-wise power with clamped exponent to avoid overflow."""
    y_clamped = torch.clamp(y, -max_exp, max_exp)
    return torch.pow(torch.abs(x) + 1e-6, y_clamped)


def log(x, _unused=None, eps: float = 1e-8):
    """Element-wise natural log with epsilon for stability."""
    return torch.log(torch.abs(x) + eps)


def max_op(x, y):
    """Element-wise maximum."""
    return torch.max(x, y)


def min_op(x, y):
    """Element-wise minimum."""
    return torch.min(x, y)


op_funcs = [
    add,
    identity,
    multiply,
    subtract,
    divide,
    power,
    log,
    max_op,
    min_op,
]


class DAGController(nn.Module):
    """Selects inputs from previous nodes and operation via attention."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.op_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.op_selector = nn.Linear(hidden_dim, len(op_funcs))
        self.last_attn: torch.Tensor | None = None
        self.last_op_weights: torch.Tensor | None = None

    def forward(
        self,
        nodes: torch.Tensor,
        operand_ctx: torch.Tensor,
        op_ctx: torch.Tensor,
    ):
        """Select two inputs and operation weights for the next DAG node.

        Args:
            nodes: Tensor of shape ``(batch, num_nodes, hidden_dim)``.
            operand_ctx: Context embedding for operand selection.
            op_ctx: Context embedding for operation selection.
        """
        summary = torch.mean(nodes, dim=1)
        query1 = self.query_proj1(summary) + operand_ctx
        query2 = self.query_proj2(summary) + operand_ctx
        keys = self.key_proj(nodes)
        att1 = torch.einsum("bij,bj->bi", keys, query1) / (nodes.size(-1) ** 0.5)
        attn1 = F.softmax(att1, dim=1)
        input1 = torch.sum(attn1.unsqueeze(-1) * nodes, dim=1)
        att2 = torch.einsum("bij,bj->bi", keys, query2) / (nodes.size(-1) ** 0.5)
        attn2 = F.softmax(att2, dim=1)
        input2 = torch.sum(attn2.unsqueeze(-1) * nodes, dim=1)

        op_query = self.op_query_proj(summary) + op_ctx
        att_op = torch.einsum("bij,bj->bi", keys, op_query) / (nodes.size(-1) ** 0.5)
        attn_op = F.softmax(att_op, dim=1)
        op_context = torch.sum(attn_op.unsqueeze(-1) * nodes, dim=1)
        op_logits = self.op_selector(op_context)
        op_weights = F.softmax(op_logits, dim=-1)

        self.last_attn = attn1.detach()
        self.last_op_weights = op_weights.detach()
        return input1, input2, op_weights


class DifferentiableDAG(nn.Module):
    """Engine that grows a differentiable DAG of intermediate nodes."""

    def __init__(self, hidden_dim, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.controller = DAGController(hidden_dim)
        self.step_emb = nn.Embedding(num_steps, hidden_dim)

    def forward(
        self,
        initial_nodes,
        operand_ctx: torch.Tensor,
        op_ctx: torch.Tensor,
        return_info: bool = False,
    ):
        """Build the DAG and return all nodes.

        Args:
            initial_nodes: Sequence of ``(batch, hidden_dim)`` tensors.
            return_info: If True, also return attention and op weight history.
        """
        nodes = list(initial_nodes)
        attn_history = []
        op_history = []
        step_embs = self.step_emb.weight[: self.num_steps]
        for step in range(self.num_steps):
            node_tensor = torch.stack(nodes, dim=1)
            emb = step_embs[step].unsqueeze(0).expand(node_tensor.size(0), -1)
            input1, input2, op_weights = self.controller(
                node_tensor,
                operand_ctx + emb,
                op_ctx + emb,
            )
            if return_info:
                attn_history.append(self.controller.last_attn)
                op_history.append(self.controller.last_op_weights)
            outs = [op(input1, input2) for op in op_funcs]
            stack = torch.stack(outs, dim=1)
            new_node = torch.sum(op_weights.unsqueeze(-1) * stack, dim=1)
            nodes.append(new_node)
        stacked = torch.stack(nodes, dim=1)
        if return_info:
            return stacked, attn_history, op_history
        return stacked


# -----------------------------------------------------------------------------
# GPT wrapper with DAG
# -----------------------------------------------------------------------------
from dataclasses import dataclass

from model import GPT, Block, GPTConfig


@dataclass
class DAGGPTConfig(GPTConfig):
    dag_depth: int = 4  # Number of DAG steps to perform


class DAGGPT(GPT):
    """GPT model augmented with differentiable DAG reasoning."""

    def __init__(self, config: DAGGPTConfig):
        super().__init__(config)
        # initialize DAG components
        self.dag = DifferentiableDAG(config.n_embd, config.dag_depth)
        self.mix_gate = nn.Linear(config.n_embd * 2, 1)
        self.token_attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )
        self.attn_to_num = nn.Linear(config.n_embd, 1)
        self.snap_block = Block(config)
        self.operand_ctx_block = Block(config)
        self.op_ctx_block = Block(config)
        self.post_dag_block = Block(config)

    @classmethod
    def init_from(
        cls, model_type: str, override_args: Optional[dict] = None
    ) -> "DAGGPT":
        """Initialize a DAGGPT model from a pretrained GPT model.

        Args:
            model_type: Type of pretrained model to load (e.g. 'gpt2')
            override_args: Optional dict of arguments to override in the config

        Returns:
            Initialized DAGGPT model
        """
        # First initialize the base GPT model
        base_model = GPT.from_pretrained(model_type, override_args)

        # Create DAGGPT config from base model config
        config = DAGGPTConfig(**base_model.config.__dict__)
        if override_args:
            for k, v in override_args.items():
                setattr(config, k, v)

        # Create new DAGGPT model
        model = cls(config)

        # Copy over the base model weights
        model.load_state_dict(base_model.state_dict(), strict=False)

        return model

    # Expected dimensions:
    # idx: (batch_size, seq_len)
    # targets: (batch_size, seq_len)
    def forward(self, idx, targets=None, return_dag_info: bool = False):
        """Run the model and optionally return DAG attention info."""
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        emb = tok_emb + pos_emb
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        hidden = self.transformer.ln_f(x)

        snap_hidden = self.snap_block(hidden)
        attn_out, _ = self.token_attn(snap_hidden, snap_hidden, snap_hidden)
        operand_ctx = self.operand_ctx_block(hidden).mean(dim=1)
        op_ctx = self.op_ctx_block(hidden).mean(dim=1)
        all_vals = self.attn_to_num(attn_out).squeeze(-1)
        all_nodes = all_vals.unsqueeze(-1).expand(-1, -1, tok_emb.size(-1))
        initial_nodes = [all_nodes[:, i, :] for i in range(t)]
        zero_node = torch.zeros_like(tok_emb[:, 0, :])
        while len(initial_nodes) < 2:
            initial_nodes.append(zero_node)

        # Store activations for entropy calculation
        self.last_activations = {
            "snap_hidden": snap_hidden,
            "attn_out": attn_out,
            "operand_ctx": operand_ctx,
            "op_ctx": op_ctx,
            "all_nodes": all_nodes,
        }
        for tensor in self.last_activations.values():
            tensor.retain_grad()

        dag_result = self.dag(
            initial_nodes, operand_ctx, op_ctx, return_info=return_dag_info
        )
        if return_dag_info:
            dag_nodes, attn_hist, op_hist = dag_result
        else:
            dag_nodes = dag_result
            attn_hist = op_hist = None
        dag_output = dag_nodes[:, -1, :]
        dag_sem = self.post_dag_block(dag_output.unsqueeze(1)).squeeze(1)
        gate = torch.sigmoid(
            self.mix_gate(torch.cat([hidden[:, -1, :], dag_sem], dim=-1))
        )
        new_last = (1 - gate) * hidden[:, -1, :] + gate * dag_sem
        hidden = torch.cat([hidden[:, :-1, :], new_last.unsqueeze(1)], dim=1)
        logits = self.lm_head(hidden)
        if targets is not None:
            loss = self._compute_loss(logits, targets)
        else:
            loss = None
        if return_dag_info:
            return logits, loss, dag_output, {"attn": attn_hist, "op": op_hist}
        return logits, loss

    def extra_vals(self):
        """Calculate entropy and gradients of activations in the DAG components."""
        if not hasattr(self, "last_activations"):
            return {}

        entropy_vals = {}
        grad_vals = {}
        for name, tensor in self.last_activations.items():
            # Calculate entropy along the last dimension
            probs = F.softmax(tensor, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # Average over batch and sequence dimensions
            entropy_vals[f"dag_entropy/{name}"] = entropy.mean().item()

            # Get mean absolute gradient if available
            grad = getattr(tensor, "grad", None)
            if grad is not None:
                grad_vals[f"dag_grad/{name}"] = grad.abs().mean().item()
            else:
                grad_vals[f"dag_grad/{name}"] = None

        return {**entropy_vals, **grad_vals}
