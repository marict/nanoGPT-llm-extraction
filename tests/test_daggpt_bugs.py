import sys
from pathlib import Path

import pytest
import torch

# allow "import dag_model" from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dag_model
from dag_model import GPT, DAGController, GPTConfig, op_funcs


# ---------------------------------------------------------------------------
# Controller behaviour
# ---------------------------------------------------------------------------
def test_controller_selects_distinct_inputs():
    torch.manual_seed(0)
    controller = DAGController(
        hidden_dim=4, n_ops=len(op_funcs), temperature=2.0
    )  # Use standard temp

    nodes = torch.randn(1, 3, 4)  # (B, N, H)
    ctx = torch.zeros(1, 4)  # (B, H)

    att1, att2, _ = controller(nodes, ctx, ctx)

    # With Gumbel softmax, both att1 and att2 should sum to 1
    assert torch.allclose(att1.sum(dim=1), torch.ones(1))  # Should sum to 1
    assert torch.allclose(att2.sum(dim=1), torch.ones(1))  # Should sum to 1

    # Each should have one dominant value (not necessarily exactly one-hot)
    # Find max value and ensure it's significantly larger than others
    max_val1 = att1.max(dim=1).values
    max_val2 = att2.max(dim=1).values

    # The max value should be at least 0.5 (dominant)
    assert torch.all(max_val1 > 0.5), "att1 should have a dominant value"
    assert torch.all(max_val2 > 0.5), "att2 should have a dominant value"
    # Count significant elements (should be exactly 1 per distribution)
    assert (
        torch.sum(att1 > 0.5).item() == 1
    ), "att1 should have exactly one dominant element"
    assert (
        torch.sum(att2 > 0.5).item() == 1
    ), "att2 should have exactly one dominant element"


# ---------------------------------------------------------------------------
# Weight tying (WTE ↔ lm_head)
# ---------------------------------------------------------------------------
def test_weight_tying():
    cfg = GPTConfig(
        vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = GPT(cfg)
    wte_weight = model.transformer.wte.weight
    lm_weight = model.lm_head.weight
    # Pointers should match when tying is in effect
    assert wte_weight.data_ptr() == lm_weight.data_ptr()


# ---------------------------------------------------------------------------
# Block-size guard
# ---------------------------------------------------------------------------
def test_forward_block_size_assertion():
    cfg = GPTConfig(
        vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = GPT(cfg)
    x = torch.randint(0, 10, (1, 3))  # length 3 > block_size 2
    with pytest.raises(AssertionError):
        model(x)


# ---------------------------------------------------------------------------
# Loss is returned – and backward works
# ---------------------------------------------------------------------------
def test_forward_returns_loss_when_targets_given():
    cfg = GPTConfig(
        vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, loss = model(x, targets=x)

    assert loss is not None and loss.ndim == 0
    loss.backward()

    # Check that gradients reached tied head
    assert model.lm_head.weight.grad is not None
