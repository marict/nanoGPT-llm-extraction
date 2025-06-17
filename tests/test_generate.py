import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig


def test_generate_handles_cropping():
    cfg = GPTConfig(vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT(cfg)
    model.eval()
    torch.manual_seed(0)
    start = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(start, max_new_tokens=1)
    assert out.shape == (1, 6)
    assert torch.equal(out[:, :5], start)
    assert torch.all(out < cfg.vocab_size)
