import sys
from pathlib import Path

import pytest
import torch

# allow "import dag_model" from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dag_model
from dag_model import GPT, DAGPlanPredictor, GPTConfig, op_funcs


# ---------------------------------------------------------------------------
# Plan predictor behaviour
# ---------------------------------------------------------------------------
def test_plan_predictor_valid_outputs():
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=20,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=2,
        dag_scratch_nodes=2,
    )

    plan_predictor = DAGPlanPredictor(config, temperature=1.0)

    # Test batch of hidden states
    hidden_states = torch.randn(1, 3, 8)  # (B, T, H)

    op1_probs, op2_probs, op_probs = plan_predictor(hidden_states)

    # Check shapes
    B, T, depth, max_nodes = op1_probs.shape
    assert op1_probs.shape == (1, 3, 2, 8)  # B, T, dag_depth, max_nodes_per_token
    assert op2_probs.shape == (1, 3, 2, 8)  # B, T, dag_depth, max_nodes_per_token
    assert op_probs.shape == (1, 3, 2, len(op_funcs))  # B, T, dag_depth, n_ops

    # Check that probabilities sum to 1 across appropriate dimensions
    assert torch.allclose(op1_probs.sum(dim=-1), torch.ones(1, 3, 2))
    assert torch.allclose(op2_probs.sum(dim=-1), torch.ones(1, 3, 2))
    assert torch.allclose(op_probs.sum(dim=-1), torch.ones(1, 3, 2))

    # Check that all probabilities are non-negative and finite
    assert torch.all(op1_probs >= 0), "op1_probs should be non-negative"
    assert torch.all(op2_probs >= 0), "op2_probs should be non-negative"
    assert torch.all(op_probs >= 0), "op_probs should be non-negative"
    assert torch.all(torch.isfinite(op1_probs)), "op1_probs should be finite"
    assert torch.all(torch.isfinite(op2_probs)), "op2_probs should be finite"
    assert torch.all(torch.isfinite(op_probs)), "op_probs should be finite"


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
