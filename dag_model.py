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
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.op_selector = nn.Linear(hidden_dim, num_ops)
        self.last_attn: torch.Tensor | None = None
        self.last_op_weights: torch.Tensor | None = None

    def forward(self, nodes):
        """Select two inputs and operation weights for the next DAG node."""
        query = self.query_proj(nodes[-1])
        keys = self.key_proj(nodes[:-1])
        att = torch.matmul(keys, query) / (nodes.size(1) ** 0.5)
        attn = F.softmax(att, dim=0)
        input1 = torch.sum(attn.unsqueeze(-1) * nodes[:-1], dim=0)
        input2 = input1.clone()
        op_logits = self.op_selector(input1)
        op_weights = F.softmax(op_logits, dim=0)
        self.last_attn = attn.detach()
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
            initial_nodes: List of starting tensors.
            return_info: If True, also return attention and op weight history.
        """
        nodes = [n for n in initial_nodes]
        attn_history = []
        op_history = []
        for _ in range(self.num_steps):
            node_tensor = torch.stack(nodes)
            input1, input2, op_weights = self.controller(node_tensor)
            if return_info:
                attn_history.append(self.controller.last_attn)
                op_history.append(self.controller.last_op_weights)
            results = []
            for i, op in enumerate(op_funcs):
                out = op(input1, input2)
                results.append(op_weights[i] * out)
            new_node = sum(results)
            nodes.append(new_node)
        stacked = torch.stack(nodes)
        if return_info:
            return stacked, attn_history, op_history
        return stacked


# -----------------------------------------------------------------------------
# GPT wrapper with DAG
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from model import GPT, GPTConfig
from binary_embedding import BinaryAwareEmbedding


@dataclass
class DAGGPTConfig(GPTConfig):
    dag_depth: int = 4
    dag_hidden_dim: int = 16
    dag_num_ops: int = 5


class DAGGPT(GPT):
    """GPT model augmented with differentiable DAG reasoning."""

    def __init__(self, config: DAGGPTConfig):
        super().__init__(config)
        # replace the token embedding with binary-aware embedding
        self.transformer.wte = BinaryAwareEmbedding(config.vocab_size, config.n_embd)
        self.transformer.wte.apply(self._init_weights)
        self.dag = DifferentiableDAG(config.n_embd, config.dag_num_ops, config.dag_depth)

    def forward(self, idx, binary=None, targets=None, return_dag_info: bool = False):
        """Run the model and optionally return DAG attention info."""
        if binary is None:
            binary = torch.zeros(idx.size(0), idx.size(1), 33, device=idx.device)
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx, binary)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        hidden = self.transformer.ln_f(x)
        logits = self.lm_head(hidden)
        loss = None
        last_hidden = hidden[:, -1, :].squeeze(0)
        dag_result = self.dag([last_hidden, last_hidden], return_info=return_dag_info)
        if return_dag_info:
            dag_nodes, attn_hist, op_hist = dag_result
        else:
            dag_nodes = dag_result
            attn_hist = op_hist = None
        dag_output = dag_nodes[-1]
        if return_dag_info:
            return logits, loss, dag_output, {"attn": attn_hist, "op": op_hist}
        return logits, loss, dag_output
