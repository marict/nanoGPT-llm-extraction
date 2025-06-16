import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dag_model
from dag_model import (
    DAGGPT,
    DAGGPTConfig,
    DifferentiableDAG,
    DAGController,
    multiply,
    subtract,
    divide,
)
import pytest


@pytest.fixture(scope="module")
def small_dag_gpt():
    config = DAGGPTConfig(vocab_size=20, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=2)
    return DAGGPT(config), config


def test_dag_gpt_forward(small_dag_gpt):
    model, config = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    logits, loss, dag_out = model(x)
    assert logits.shape == (2, 4, 20)
    assert dag_out.shape == (2, config.n_embd)


def test_dag_node_growth_regression(monkeypatch):
    class DummyController(DAGController):
        def forward(self, nodes):
            input1 = nodes[:, -1, :]
            input2 = input1
            op_weights = torch.tensor([[1.0, 0.0]])
            return input1, input2, op_weights

    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    dag = DifferentiableDAG(hidden_dim=4, num_ops=2, num_steps=2)
    dag.controller = DummyController(4, 2)
    out = dag([torch.ones(1, 4)])
    assert out.shape == (1, 3, 4)
    assert torch.allclose(out[:, -1, :], torch.full((1, 4), 4.0))


def test_op_functions():
    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([1.0, 2.0])
    assert torch.allclose(multiply(x, y), x * y)
    assert torch.allclose(subtract(x, y), x - y)
    assert torch.allclose(divide(x, y), x / y)


def test_dag_backward_flow(small_dag_gpt):
    model, _ = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    _, _, dag_out = model(x)
    loss = dag_out.sum()
    loss.backward()
    grad = model.dag.controller.op_selector.weight.grad
    assert grad is not None




def test_dag_initial_nodes_all_tokens(monkeypatch):
    tokens = [0, 1, 2, 3]

    cfg = DAGGPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=1,
    )
    model = DAGGPT(cfg)

    class DummyAttn(nn.Module):
        def forward(self, q, k, v):
            return v, None

    class DummyProj(nn.Module):
        def forward(self, x):
            return torch.full((*x.shape[:-1], 1), 3.0)

    model.token_attn = DummyAttn()
    model.attn_to_num = DummyProj()

    captured = {}
    original_forward = DifferentiableDAG.forward

    def capture_forward(self, initial_nodes, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(self, initial_nodes, return_info=return_info)

    monkeypatch.setattr(DifferentiableDAG, "forward", capture_forward)

    x = torch.tensor(tokens).unsqueeze(0)
    model(x)

    assert "nodes" in captured
    init_nodes = captured["nodes"]
    assert len(init_nodes) == len(tokens)
    expected = torch.full((len(tokens), 8), 3.0)
    for node, exp in zip(init_nodes, expected):
        assert torch.allclose(node.squeeze(0), exp)


def test_dag_attention_for_non_numeric(monkeypatch):
    tokens = [5, 6, 7]

    cfg = DAGGPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=1,
    )
    model = DAGGPT(cfg)

    class DummyAttn(nn.Module):
        def forward(self, q, k, v):
            return v, None

    class DummyProj(nn.Module):
        def forward(self, x):
            vals = torch.arange(1, x.size(1) + 1, dtype=torch.float32)
            return vals.view(1, -1, 1).expand_as(x[:, :, :1])

    model.token_attn = DummyAttn()
    model.attn_to_num = DummyProj()

    captured = {}
    original_forward = DifferentiableDAG.forward

    def capture_forward(self, initial_nodes, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(self, initial_nodes, return_info=return_info)

    monkeypatch.setattr(DifferentiableDAG, "forward", capture_forward)

    x = torch.tensor(tokens).unsqueeze(0)
    model(x)

    init_nodes = captured["nodes"]
    assert len(init_nodes) == len(tokens)
    expected0 = torch.full((8,), 1.0)
    expected1 = torch.full((8,), 2.0)
    expected2 = torch.full((8,), 3.0)
    assert torch.allclose(init_nodes[0].squeeze(0), expected0)
    assert torch.allclose(init_nodes[1].squeeze(0), expected1)
    assert torch.allclose(init_nodes[2].squeeze(0), expected2)


def test_zero_padding_single_token(monkeypatch):
    tokens = [7]
    assert len(tokens) == 1

    cfg = DAGGPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=1,
    )
    model = DAGGPT(cfg)

    captured = {}
    original_forward = DifferentiableDAG.forward

    def capture_forward(self, initial_nodes, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(self, initial_nodes, return_info=return_info)

    monkeypatch.setattr(DifferentiableDAG, "forward", capture_forward)

    x = torch.tensor(tokens).unsqueeze(0)
    model(x)

    init_nodes = captured["nodes"]
    assert len(init_nodes) == 2
    assert init_nodes[1].squeeze(0).abs().sum() == 0


def test_post_dag_block_called(monkeypatch):
    cfg = DAGGPTConfig(vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1)
    model = DAGGPT(cfg)

    called = {}

    def fake_forward(self, x):
        called["used"] = True
        return x

    monkeypatch.setattr(model.post_dag_block, "forward", fake_forward.__get__(model.post_dag_block, type(model.post_dag_block)))

    x = torch.randint(0, 10, (1, 2))
    model(x)
    assert called.get("used")
