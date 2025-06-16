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


op_funcs = [add, identity, multiply, subtract, divide]


class DAGController(nn.Module):
    """Selects inputs from previous nodes and operation via attention."""

    def __init__(self, hidden_dim, num_ops):
        super().__init__()
        self.query_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.op_selector = nn.Linear(hidden_dim, num_ops)
        self.last_attn: torch.Tensor | None = None
        self.last_op_weights: torch.Tensor | None = None

    def forward(self, nodes: torch.Tensor):
        """Select two inputs and operation weights for the next DAG node.

        Args:
            nodes: Tensor of shape ``(batch, num_nodes, hidden_dim)``.
        """
        query1 = self.query_proj1(nodes[:, -1, :])
        query2 = self.query_proj2(nodes[:, -1, :])
        keys = self.key_proj(nodes[:, :-1, :])
        att1 = torch.einsum("bij,bj->bi", keys, query1) / (nodes.size(-1) ** 0.5)
        attn1 = F.softmax(att1, dim=1)
        input1 = torch.sum(attn1.unsqueeze(-1) * nodes[:, :-1, :], dim=1)
        att2 = torch.einsum("bij,bj->bi", keys, query2) / (nodes.size(-1) ** 0.5)
        attn2 = F.softmax(att2, dim=1)
        input2 = torch.sum(attn2.unsqueeze(-1) * nodes[:, :-1, :], dim=1)
        op_logits = self.op_selector(input1)
        op_weights = F.softmax(op_logits, dim=-1)
        self.last_attn = attn1.detach()
        self.last_op_weights = op_weights.detach()
        return input1, input2, op_weights


class DifferentiableDAG(nn.Module):
    """Engine that grows a differentiable DAG of intermediate nodes."""

    def __init__(self, hidden_dim, num_ops, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.controller = DAGController(hidden_dim, num_ops)

    def forward(self, initial_nodes, return_info: bool = False):
        """Build the DAG and return all nodes.

        Args:
            initial_nodes: Sequence of ``(batch, hidden_dim)`` tensors.
            return_info: If True, also return attention and op weight history.
        """
        nodes = [n for n in initial_nodes]
        attn_history = []
        op_history = []
        for _ in range(self.num_steps):
            node_tensor = torch.stack(nodes, dim=1)
            input1, input2, op_weights = self.controller(node_tensor)
            if return_info:
                attn_history.append(self.controller.last_attn)
                op_history.append(self.controller.last_op_weights)
            results = []
            for i, op in enumerate(op_funcs):
                out = op(input1, input2)
                results.append(op_weights[:, i].unsqueeze(-1) * out)
            new_node = sum(results)
            nodes.append(new_node)
        stacked = torch.stack(nodes, dim=1)
        if return_info:
            return stacked, attn_history, op_history
        return stacked


# -----------------------------------------------------------------------------
# GPT wrapper with DAG
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from model import GPT, GPTConfig, Block


@dataclass
class DAGGPTConfig(GPTConfig):
    dag_depth: int = 4
    dag_hidden_dim: int = 16
    dag_num_ops: int = 5


class DAGGPT(GPT):
    """GPT model augmented with differentiable DAG reasoning."""

    def __init__(self, config: DAGGPTConfig):
        super().__init__(config)
        # initialize DAG components
        self.dag = DifferentiableDAG(config.n_embd, config.dag_num_ops, config.dag_depth)
        self.mix_gate = nn.Linear(config.n_embd * 2, 1)
        self.token_attn = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)
        self.attn_to_num = nn.Linear(config.n_embd, 1)
        self.snap_block = Block(config)
        self.post_dag_block = Block(config)

    def forward(self, idx, targets=None, return_dag_info: bool = False):
        """Run the model and optionally return DAG attention info."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        emb = tok_emb + pos_emb
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        hidden = self.transformer.ln_f(x)
        loss = None

        snap_hidden = self.snap_block(hidden)
        attn_out, _ = self.token_attn(snap_hidden, snap_hidden, snap_hidden)
        all_vals = torch.round(self.attn_to_num(attn_out).squeeze(-1))
        all_nodes = all_vals.unsqueeze(-1).expand(-1, -1, tok_emb.size(-1))
        initial_nodes = [all_nodes[:, i, :] for i in range(t)]
        zero_node = torch.zeros_like(tok_emb[:, 0, :])
        while len(initial_nodes) < 2:
            initial_nodes.append(zero_node)

        dag_result = self.dag(initial_nodes, return_info=return_dag_info)
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
        hidden = hidden.clone()
        hidden[:, -1, :] = (1 - gate) * hidden[:, -1, :] + gate * dag_sem
        logits = self.lm_head(hidden)
        if return_dag_info:
            return logits, loss, dag_output, {"attn": attn_hist, "op": op_hist}
        return logits, loss, dag_output
