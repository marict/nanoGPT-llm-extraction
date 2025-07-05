import pytest
import torch

from dag_model import GPT, GPTConfig, _rms_rescale


def test_torch_compile_forward():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    cfg = GPTConfig(dag_depth=2, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = GPT(cfg)
    compiled = torch.compile(model, mode="reduce-overhead")
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, _ = compiled(idx)
    assert torch.isfinite(logits).all()


def test_torch_compile_forward_backward():
    """Ensure compiled model can run a backward pass without errors and produces finite grads."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    cfg = GPTConfig(
        dag_depth=2,
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=16,
        vocab_size=128,  # small vocab avoids scatter kernel bug on some CPUs
    )
    model = GPT(cfg)
    compiled = torch.compile(model, mode="reduce-overhead")

    # Dummy batch
    B = 2
    idx = torch.randint(0, cfg.vocab_size, (B, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (B, cfg.block_size))

    logits, _ = compiled(idx, targets)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )
    loss.backward()

    # Check one representative parameter has finite gradient
    grad_ok = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            grad_ok = True
            break

    assert grad_ok, "No parameter received a gradient"


def test_rms_rescale_compile():
    def fn(x):
        # Build a simple (B,T,S) buffer with S=2
        buf = torch.stack([x.clone(), x.clone()], dim=-1)
        _rms_rescale(buf, 1)
        return buf[..., 1]

    x = torch.randn(2, 3)
    compiled_fn = torch.compile(fn, mode="reduce-overhead")
    out = compiled_fn(x)
    assert torch.isfinite(out).all()
