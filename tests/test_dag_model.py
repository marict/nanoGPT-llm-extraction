import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

import dag_model
from dag_model import (DAGGPT, DAGController, DAGGPTConfig, DifferentiableDAG,
                       divide, multiply, subtract)


@pytest.fixture(scope="module")
def small_dag_gpt():
    config = DAGGPTConfig(
        vocab_size=20, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=2
    )
    return DAGGPT(config), config


def test_dag_gpt_forward(small_dag_gpt):
    model, config = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    logits, loss, dag_out = model(x)
    assert logits.shape == (2, 4, 20)
    assert dag_out.shape == (2, config.n_embd)


def test_dag_node_growth_regression(monkeypatch):
    class DummyController(DAGController):
        def forward(self, nodes, operand_ctx, op_ctx):
            input1 = nodes[:, -1, :]
            input2 = input1
            op_weights = torch.tensor([[1.0, 0.0]])
            return input1, input2, op_weights

    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    dag = DifferentiableDAG(hidden_dim=4, num_steps=2)
    dag.controller = DummyController(4)
    ctx = torch.zeros(1, 4)
    out = dag([torch.ones(1, 4)], ctx, ctx)
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

    def capture_forward(self, initial_nodes, operand_ctx, op_ctx, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(
            self, initial_nodes, operand_ctx, op_ctx, return_info=return_info
        )

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

    def capture_forward(self, initial_nodes, operand_ctx, op_ctx, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(
            self, initial_nodes, operand_ctx, op_ctx, return_info=return_info
        )

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

    def capture_forward(self, initial_nodes, operand_ctx, op_ctx, return_info=False):
        captured["nodes"] = initial_nodes
        return original_forward(
            self, initial_nodes, operand_ctx, op_ctx, return_info=return_info
        )

    monkeypatch.setattr(DifferentiableDAG, "forward", capture_forward)

    x = torch.tensor(tokens).unsqueeze(0)
    model(x)

    init_nodes = captured["nodes"]
    assert len(init_nodes) == 2
    assert init_nodes[1].squeeze(0).abs().sum() == 0


def test_post_dag_block_called(monkeypatch):
    cfg = DAGGPTConfig(
        vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = DAGGPT(cfg)

    called = {}

    def fake_forward(self, x):
        called["used"] = True
        return x

    monkeypatch.setattr(
        model.post_dag_block,
        "forward",
        fake_forward.__get__(model.post_dag_block, type(model.post_dag_block)),
    )

    x = torch.randint(0, 10, (1, 2))
    model(x)
    assert called.get("used")


def test_step_contexts_added(monkeypatch):
    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    dag = DifferentiableDAG(hidden_dim=4, num_steps=3)

    step_vals = torch.stack([torch.full((4,), float(i)) for i in range(3)])
    dag.step_emb = nn.Embedding.from_pretrained(step_vals, freeze=True)

    captured = []

    class RecController(DAGController):
        def forward(self, nodes, operand_ctx, op_ctx):
            captured.append((operand_ctx.clone(), op_ctx.clone()))
            return nodes[:, 0, :], nodes[:, 0, :], torch.tensor([[1.0, 0.0]])

    dag.controller = RecController(4)

    init = [torch.zeros(1, 4), torch.ones(1, 4)]
    op_ctx = torch.ones(1, 4)
    operand_ctx = torch.zeros(1, 4)
    dag(init, operand_ctx, op_ctx)

    assert len(captured) == 3
    for i, (oc, oc2) in enumerate(captured):
        expect = step_vals[i].unsqueeze(0)
        assert torch.allclose(oc, operand_ctx + expect)
        assert torch.allclose(oc2, op_ctx + expect)
        assert oc.shape == (1, 4) and oc2.shape == (1, 4)


def test_daggpt_config_creation():
    """Test that DAGGPTConfig can be created with various parameters."""
    # Test with default values
    config = DAGGPTConfig()
    assert config.dag_depth == 4
    assert config.n_embd == 768  # default from GPTConfig

    # Test with custom values
    config = DAGGPTConfig(
        dag_depth=6,
        n_embd=512,
        n_layer=6,
        n_head=8,
        block_size=1024,
        vocab_size=50257,
    )
    assert config.dag_depth == 6
    assert config.n_embd == 512
    assert config.n_layer == 6
    assert config.n_head == 8
    assert config.block_size == 1024
    assert config.vocab_size == 50257

    # Test that invalid parameters are rejected
    with pytest.raises(TypeError):
        DAGGPTConfig(dag_hidden_dim=32)  # This should fail
