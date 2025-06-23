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
        def __init__(self, hidden_dim, n_ops, temperature=1.0):
            super().__init__(hidden_dim, n_ops, temperature)

        def forward(self, embeds, operand_ctx, op_ctx):  # new signature
            B, N, _ = embeds.shape
            att_last = torch.zeros(B, N, device=embeds.device)
            att_last[:, -1] = 1.0  # pick last twice
            op_weights = torch.tensor([[1.0, 0.0]], device=embeds.device)  # use 'add'
            return att_last, att_last, op_weights

    dag = DifferentiableDAG(
        hidden_dim=H, num_steps=2, scalar_to_embed=proj, temperature=1.0
    )
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

    # --- stub value-extractor so each token's value == 3 -------------
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
        def __init__(self, hidden_dim, n_ops, temperature=1.0):
            super().__init__(hidden_dim, n_ops, temperature)

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
    """Test DAGGPT's extra_vals returns entropy and gradient information."""
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    # Test that extra_vals returns empty dict before any forward pass
    extra_before = model.extra_vals()
    assert isinstance(extra_before, dict)
    assert len(extra_before) == 0

    # Forward pass to populate activations
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    # Test extra_vals after forward pass but before backward pass
    extra_after_forward = model.extra_vals()
    assert isinstance(extra_after_forward, dict)

    # Should have entropy keys but no gradient keys yet
    entropy_keys = [k for k in extra_after_forward if k.startswith("dag_entropy/")]
    grad_keys = [k for k in extra_after_forward if k.startswith("dag_grad/")]

    assert len(entropy_keys) > 0, "Expected entropy keys after forward pass"
    assert len(grad_keys) == 0, "Expected no gradient keys before backward pass"

    # Verify entropy values are reasonable
    for k in entropy_keys:
        assert isinstance(
            extra_after_forward[k], float
        ), f"Entropy value {k} is not a float"
        assert extra_after_forward[k] >= 0, f"Entropy value {k} is negative"
        assert not torch.isnan(
            torch.tensor(extra_after_forward[k])
        ), f"Entropy value {k} is NaN"

    # Backward pass to populate gradients
    loss.sum().backward()

    # Test extra_vals after backward pass
    extra_after_backward = model.extra_vals()

    # Should have both entropy and gradient keys
    entropy_keys_after = [
        k for k in extra_after_backward if k.startswith("dag_entropy/")
    ]
    grad_keys_after = [k for k in extra_after_backward if k.startswith("dag_grad/")]

    assert len(entropy_keys_after) > 0, "Expected entropy keys after backward pass"
    assert len(grad_keys_after) > 0, "Expected gradient keys after backward pass"

    # Verify all values are reasonable
    for k in entropy_keys_after + grad_keys_after:
        assert isinstance(
            extra_after_backward[k], float
        ), f"Extra value {k} is not a float"
        assert not torch.isnan(
            torch.tensor(extra_after_backward[k])
        ), f"Extra value {k} is NaN"

        # Only entropy values should be non-negative, gradients can be negative
        if k.startswith("dag_entropy/"):
            assert extra_after_backward[k] >= 0, f"Entropy value {k} is negative"


def test_extra_vals_consistency_daggpt():
    """Test that DAGGPT's extra_vals returns consistent structure across calls."""
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    # Forward and backward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    loss.sum().backward()

    # Call extra_vals multiple times
    extra_vals_1 = model.extra_vals()
    extra_vals_2 = model.extra_vals()

    # Should be identical
    assert extra_vals_1.keys() == extra_vals_2.keys()
    for k in extra_vals_1.keys():
        assert extra_vals_1[k] == extra_vals_2[k], f"Inconsistent value for key {k}"


def test_hook_behavior():
    """Test that gradient hooks are properly registered and capture gradients."""
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    # Ensure dag_grads is initially empty
    assert not hasattr(model, "dag_grads") or len(model.dag_grads) == 0

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    # Check that last_activations are populated
    assert hasattr(model, "last_activations")
    assert len(model.last_activations) > 0

    # Check that dag_grads dict exists but is still empty (no backward pass yet)
    assert hasattr(model, "dag_grads")
    assert len(model.dag_grads) == 0

    # Verify hooks are registered by checking tensors require grad
    for name, tensor in model.last_activations.items():
        if tensor.requires_grad:
            # Tensor should have a hook registered (we can't directly test this,
            # but we can verify the tensor requires grad)
            assert (
                tensor.requires_grad
            ), f"Tensor {name} should require grad for hook registration"

    # Backward pass should trigger hooks
    loss.sum().backward()

    # Now dag_grads should be populated
    assert len(model.dag_grads) > 0, "dag_grads should be populated after backward pass"

    # Verify gradient values are reasonable
    for name, grad_val in model.dag_grads.items():
        assert isinstance(grad_val, float), f"Gradient {name} is not a float"
        assert not torch.isnan(torch.tensor(grad_val)), f"Gradient {name} is NaN"


def test_hook_behavior_multiple_backward_passes():
    """Test that hooks work correctly across multiple backward passes."""
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    # First forward and backward pass
    x1 = torch.randint(0, cfg.vocab_size, (2, 8))
    y1 = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss1 = model(x1, y1)
    loss1.sum().backward()

    first_grads = model.dag_grads.copy()
    assert len(first_grads) > 0

    # Clear gradients and do a second forward pass with same data
    model.zero_grad()
    _, loss2 = model(x1, y1)
    loss2.sum().backward()

    second_grads = model.dag_grads.copy()
    assert len(second_grads) > 0

    # With same inputs, gradients should be identical (or very close)
    assert first_grads.keys() == second_grads.keys()
    for k in first_grads.keys():
        assert (
            abs(first_grads[k] - second_grads[k]) < 1e-5
        ), f"Gradient {k} differs too much between identical passes"

    # Now test with different data to ensure hooks can capture different gradients
    x3 = torch.ones_like(x1) * (cfg.vocab_size - 1)  # Very different input
    y3 = torch.zeros_like(y1)  # Very different target
    model.zero_grad()
    _, loss3 = model(x3, y3)
    loss3.sum().backward()

    third_grads = model.dag_grads.copy()
    assert len(third_grads) > 0

    # Verify that the hooks are still working by checking gradients exist
    assert third_grads.keys() == first_grads.keys()
    for k in third_grads.keys():
        assert isinstance(third_grads[k], float)
        assert not torch.isnan(torch.tensor(third_grads[k]))


def test_hook_behavior_no_grad_context():
    """Test that hooks don't interfere when in no_grad context."""
    cfg = DAGGPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = DAGGPT(cfg)

    # Forward pass in no_grad context
    with torch.no_grad():
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss = model(x)

        # Should still have last_activations
        assert hasattr(model, "last_activations")
        assert len(model.last_activations) > 0

        # dag_grads should exist but be empty
        assert hasattr(model, "dag_grads")
        assert len(model.dag_grads) == 0

        # extra_vals should work and return entropy values
        extra = model.extra_vals()
        entropy_keys = [k for k in extra if k.startswith("dag_entropy/")]
        grad_keys = [k for k in extra if k.startswith("dag_grad/")]

        assert (
            len(entropy_keys) > 0
        ), "Should have entropy values even in no_grad context"
        assert len(grad_keys) == 0, "Should have no gradient values in no_grad context"
