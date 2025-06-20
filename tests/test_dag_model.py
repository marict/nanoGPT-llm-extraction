# tests/test_dag_model.py
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# --------------------------------------------------------------------- #
# import library code
# --------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dag_model  # noqa: E402
from dag_model import (DAGGPT, DAGController, DAGGPTConfig, DifferentiableDAG,
                       divide, multiply, subtract)


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def make_dummy_proj(hidden_dim: int) -> nn.Linear:
    """Tiny 1→H projector for the new DifferentiableDAG signature."""
    return nn.Linear(1, hidden_dim, bias=False)


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_dag_gpt():
    cfg = DAGGPTConfig(
        vocab_size=20,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=2,
    )
    return DAGGPT(cfg), cfg


# --------------------------------------------------------------------- #
# basic forward / backward
# --------------------------------------------------------------------- #
def test_dag_gpt_forward(small_dag_gpt):
    model, _ = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    logits, loss = model(x)
    assert logits.shape == (2, 4, 20)
    assert loss is None


def test_dag_backward_flow(small_dag_gpt):
    model, _ = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    y = torch.randint(0, 20, (2, 4))
    _, loss = model(x, y)
    loss.sum().backward()
    # grad must have reached op-selector
    assert model.dag.controller.op_selector.weight.grad is not None


# --------------------------------------------------------------------- #
# op-function sanity
# --------------------------------------------------------------------- #
def test_op_functions():
    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([1.0, 2.0])
    assert torch.allclose(multiply(x, y), x * y)
    assert torch.allclose(subtract(x, y), x - y)
    assert torch.allclose(divide(x, y), x / y)


# --------------------------------------------------------------------- #
# DAG growth / controller behaviour
# --------------------------------------------------------------------- #
def test_dag_node_growth_regression(monkeypatch):
    """Two-step DAG with dummy controller: result should be 4 × initial value."""
    H = 4
    proj = make_dummy_proj(H)

    # overwrite op_funcs with just [add, identity] for brevity
    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])

    class DummyController(DAGController):
        def forward(self, embeds, operand_ctx, op_ctx):  # new signature
            B, N, _ = embeds.shape
            att_last = torch.zeros(B, N, device=embeds.device)
            att_last[:, -1] = 1.0  # pick last twice
            op_weights = torch.tensor([[1.0, 0.0]], device=embeds.device)  # use 'add'
            return att_last, att_last, op_weights

    dag = DifferentiableDAG(hidden_dim=H, num_steps=2, scalar_to_embed=proj)
    dag.controller = DummyController(H, len(dag_model.op_funcs))

    # initial lists (one node, scalar value = 1)
    init_emb = [torch.ones(1, H)]
    init_val = [torch.ones(1)]
    ctx = torch.zeros(1, H)

    embeds, vals = dag(init_emb, init_val, ctx, ctx)
    assert len(vals) == 3  # 1 original + 2 new
    assert torch.allclose(vals[-1], torch.full((1,), 4.0))


# ---------------------------------------------------------------------
# initial-node materialisation tests  (fixed)
# ---------------------------------------------------------------------
def test_dag_initial_nodes_all_tokens(monkeypatch):
    """Every token should contribute exactly one *initial* DAG value."""
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

    # --- stub value-extractor so each token’s value == 3 -------------
    class DummyVal(nn.Module):
        def forward(self, x):  # x : (B,T,H)
            return torch.full((x.size(0), x.size(1)), 3.0, device=x.device)

    model.value_extractor = DummyVal()

    captured = {}

    # keep original forward before monkey-patching
    _orig_forward = DifferentiableDAG.forward

    def capture(self, embeds, values, *ctx, **kw):
        # clone the list *before* the DAG mutates it
        captured["init_vals"] = [v.clone() for v in values]
        return _orig_forward(self, embeds, values, *ctx, **kw)

    monkeypatch.setattr(DifferentiableDAG, "forward", capture)

    model(torch.tensor(tokens).unsqueeze(0))

    init_vals = captured["init_vals"]
    assert len(init_vals) == len(tokens)
    for v in init_vals:
        assert torch.allclose(v.squeeze(0), torch.tensor(3.0))


# ---------------------------------------------------------------------
# single-token zero-padding        (fixed recursion)
# ---------------------------------------------------------------------
def test_zero_padding_single_token(monkeypatch):
    cfg = DAGGPTConfig(
        vocab_size=10, block_size=1, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = DAGGPT(cfg)

    captured = {}
    _orig_forward = DifferentiableDAG.forward  # save before patch

    def capture(self, embeds, values, *ctx, **kw):
        captured["vals"] = [v.clone() for v in values]
        return _orig_forward(self, embeds, values, *ctx, **kw)

    monkeypatch.setattr(DifferentiableDAG, "forward", capture)
    model(torch.tensor([7]).unsqueeze(0))

    vals = captured["vals"]
    assert len(vals) == 2  # padded to two
    # second node should be exactly zero
    assert vals[1].abs().sum().item() == 0


# --------------------------------------------------------------------- #
# post-DAG block is called
# --------------------------------------------------------------------- #
def test_post_dag_block_called(monkeypatch):
    cfg = DAGGPTConfig(vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8)
    model = DAGGPT(cfg)

    called = {}

    def mark(_, x):
        called["hit"] = True
        return x

    monkeypatch.setattr(
        model.post_dag_block,
        "forward",
        mark.__get__(model.post_dag_block, type(model.post_dag_block)),
    )
    model(torch.randint(0, 10, (1, 2)))
    assert called.get("hit", False)


# ---------------------------------------------------------------------
# step-embedding context test  (expect both +step)
# ---------------------------------------------------------------------
def test_step_contexts_added(monkeypatch):
    H = 4
    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    proj = nn.Linear(1, H, bias=False)
    dag = DifferentiableDAG(hidden_dim=H, num_steps=3, scalar_to_embed=proj)

    step_vals = torch.stack([torch.full((H,), float(i)) for i in range(3)])
    dag.step_emb = nn.Embedding.from_pretrained(step_vals, freeze=True)

    captured = []

    class RecController(DAGController):
        def forward(self, embeds, operand_ctx, op_ctx):
            captured.append((operand_ctx.clone(), op_ctx.clone()))
            B, N, _ = embeds.shape
            att = torch.zeros(B, N, device=embeds.device)
            att[:, 0] = 1
            op_weights = torch.tensor([[1.0, 0.0]], device=embeds.device)
            return att, att, op_weights

    dag.controller = RecController(H, len(dag_model.op_funcs))

    init_emb = [torch.zeros(1, H), torch.ones(1, H)]
    init_val = [torch.zeros(1), torch.ones(1)]
    base_operand = torch.zeros(1, H)
    base_op = torch.ones(1, H)
    dag(init_emb, init_val, base_operand, base_op)

    assert len(captured) == 3, "Expected 3 steps, got %d" % len(captured)
    for i, (oc1, oc2) in enumerate(captured):
        step = step_vals[i].unsqueeze(0)
        assert torch.allclose(oc1, base_operand + step, atol=1e-5, rtol=0), (
            "Step %d: operand context mismatch" % i
        )
        assert torch.allclose(oc2, base_op + step, atol=1e-5, rtol=0), (
            "Step %d: op context mismatch" % i
        )


# --------------------------------------------------------------------- #
# config & extra-vals
# --------------------------------------------------------------------- #
def test_daggpt_config_creation():
    cfg = DAGGPTConfig()
    assert cfg.dag_depth == 4

    cfg2 = DAGGPTConfig(
        dag_depth=6, n_embd=512, n_layer=6, n_head=8, block_size=1024, vocab_size=50257
    )
    assert cfg2.dag_depth == 6 and cfg2.n_embd == 512

    with pytest.raises(TypeError):
        DAGGPTConfig(dag_hidden_dim=32)  # invalid kwarg


# ---------------------------------------------------------------------
# extra-vals entropy / grad check  (robust to dimensionality)
# ---------------------------------------------------------------------
def test_extra_vals_daggpt():
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    loss.sum().backward()

    extra = model.extra_vals()

    # entropy & grad keys exist
    entropy_keys = [k for k in extra if k.startswith("dag_entropy/")]
    grad_keys = [k for k in extra if k.startswith("dag_grad/")]
    assert entropy_keys and grad_keys, "Expected entropy and grad keys"
    for k in entropy_keys + grad_keys:
        assert isinstance(extra[k], float), "Extra value %s is not a float" % k
        assert extra[k] >= 0, "Extra value %s is negative" % k
