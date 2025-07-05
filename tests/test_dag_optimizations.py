import torch

from dag_model import GPT, GPTConfig, add_log_space, subtract_log_space


def _finite_and_nonzero(t: torch.Tensor):
    return torch.isfinite(t).all() and t.abs().sum().item() > 0


def test_add_log_space_grad():
    sx = torch.tensor([1.0])
    sy = torch.tensor([-1.0])
    lx = torch.tensor([2.0], requires_grad=True)
    ly = torch.tensor([1.5], requires_grad=True)
    s_out, l_out = add_log_space(sx, lx, sy, ly)
    # simple scalar loss
    loss = (s_out * l_out).sum()
    loss.backward()
    assert _finite_and_nonzero(lx.grad)
    assert _finite_and_nonzero(ly.grad)


def test_subtract_log_space_grad():
    sx = torch.tensor([1.0])
    sy = torch.tensor([1.0])
    lx = torch.tensor([2.0], requires_grad=True)
    ly = torch.tensor([1.2], requires_grad=True)
    s_out, l_out = subtract_log_space(sx, lx, sy, ly)
    loss = (s_out * l_out).sum()
    loss.backward()
    assert _finite_and_nonzero(lx.grad)
    assert _finite_and_nonzero(ly.grad)


def test_dag_forward_no_nan():
    torch.manual_seed(0)
    cfg = GPTConfig(dag_depth=2, n_layer=2, n_head=2, n_embd=128, block_size=32)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = model(idx)
    assert torch.isfinite(logits).all()
