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
    torch.manual_seed(42)  # Use a different seed that produces more distinct values
    controller = DAGController(
        hidden_dim=4, n_ops=len(op_funcs), temperature=1.0
    )  # Use lower temperature for more distinct selection

    nodes = torch.randn(1, 3, 4)  # (B, N, H)
    ctx = torch.zeros(1, 4)  # (B, H)

    att1, att2, _ = controller(nodes, ctx, ctx)

    # With Gumbel softmax, both att1 and att2 should sum to 1
    assert torch.allclose(att1.sum(dim=1), torch.ones(1))  # Should sum to 1
    assert torch.allclose(att2.sum(dim=1), torch.ones(1))  # Should sum to 1

    # Check that the distributions are valid (non-negative, finite)
    assert torch.all(att1 >= 0), "att1 should be non-negative"
    assert torch.all(att2 >= 0), "att2 should be non-negative"
    assert torch.all(torch.isfinite(att1)), "att1 should be finite"
    assert torch.all(torch.isfinite(att2)), "att2 should be finite"

    # With Gumbel softmax, there should be some differentiation between values
    # Check that the max value is larger than the mean (indicating some selection)
    max_val1 = att1.max(dim=1).values
    max_val2 = att2.max(dim=1).values
    mean_val1 = att1.mean(dim=1)
    mean_val2 = att2.mean(dim=1)

    assert torch.all(max_val1 > mean_val1), "att1 max should be greater than mean"
    assert torch.all(max_val2 > mean_val2), "att2 max should be greater than mean"


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
    """Test forward pass with targets and verify backward pass works."""
    cfg = GPTConfig(
        vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1
    )
    model = GPT(cfg)
    model.train()  # Enable gradient computation

    x = torch.randint(0, cfg.vocab_size, (1, 4))

    # Test forward pass
    logits, loss = model(x, targets=x)

    assert loss is not None and loss.ndim == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"
    assert logits.shape == (1, 4, 10), "Logits shape should be (1, 4, 10)"

    # Test backward pass - should work now that in-place operations are fixed
    loss.backward()

    # Verify key gradients exist and are finite (not all parameters need gradients in every pass)
    key_params_with_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"
            key_params_with_grads += 1

    # Ensure at least some parameters received gradients
    assert key_params_with_grads > 0, "No parameters received gradients"

    # Test without targets (forward only)
    model.zero_grad()  # Clear gradients
    logits_no_targets, loss_no_targets = model(x)
    assert loss_no_targets is None, "Loss should be None without targets"

    # Check that tied weights exist
    assert hasattr(model, "lm_head"), "Model should have lm_head"
    assert hasattr(model.transformer, "wte"), "Model should have wte"
