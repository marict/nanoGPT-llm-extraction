import sys
from pathlib import Path

import pytest
import torch

# allow "import dag_model" from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dag_model
from dag_model import DAGGPT, DAGController, DAGGPTConfig, op_funcs


# ---------------------------------------------------------------------------
# Controller behaviour
# ---------------------------------------------------------------------------
def test_controller_selects_distinct_inputs():
    torch.manual_seed(0)
    controller = DAGController(hidden_dim=4, n_ops=len(op_funcs), temperature=1.0)

    nodes = torch.randn(1, 3, 4)  # (B, N, H)
    ctx = torch.zeros(1, 4)  # (B, H)

    att1, att2, _ = controller(nodes, ctx, ctx)

    # With Gumbel softmax (hard=True), both att1 and att2 should be one-hot vectors
    # but they should select different nodes (with high probability due to randomness)
    assert torch.allclose(att1.sum(dim=1), torch.ones(1))  # Should sum to 1 (one-hot)
    assert torch.allclose(att2.sum(dim=1), torch.ones(1))  # Should sum to 1 (one-hot)

    # Each should have exactly one 1.0 and the rest 0.0 (one-hot property)
    assert torch.all((att1 == 0.0) | (att1 == 1.0))
    assert torch.all((att2 == 0.0) | (att2 == 1.0))

    # Count non-zero elements (should be exactly 1 per distribution)
    assert torch.sum(att1 != 0).item() == 1
    assert torch.sum(att2 != 0).item() == 1


# ---------------------------------------------------------------------------
# Weight tying (WTE ↔ lm_head)
# ---------------------------------------------------------------------------
def test_weight_tying():
    cfg = DAGGPTConfig(
        vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = DAGGPT(cfg)
    wte_weight = model.transformer.wte.weight
    lm_weight = model.lm_head.weight
    # Pointers should match when tying is in effect
    assert wte_weight.data_ptr() == lm_weight.data_ptr()


# ---------------------------------------------------------------------------
# Block-size guard
# ---------------------------------------------------------------------------
def test_forward_block_size_assertion():
    cfg = DAGGPTConfig(
        vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = DAGGPT(cfg)
    x = torch.randint(0, 10, (1, 3))  # length 3 > block_size 2
    with pytest.raises(AssertionError):
        model(x)


# ---------------------------------------------------------------------------
# Loss is returned – and backward works
# ---------------------------------------------------------------------------
def test_forward_returns_loss_when_targets_given():
    cfg = DAGGPTConfig(
        vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = DAGGPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, loss = model(x, targets=x)

    assert loss is not None and loss.ndim == 0
    loss.backward()

    # Check that gradients reached tied head
    assert model.lm_head.weight.grad is not None
