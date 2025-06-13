import torch
import torch.nn as nn


class BinaryAwareEmbedding(nn.Module):
    """Embedding that appends numeric tag information."""

    def __init__(self, vocab_size: int, embed_dim: int, binary_dim: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.binary_dim = binary_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.binary_proj = nn.Linear(binary_dim + 1, embed_dim)

    def forward(self, tokens: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
        base = self.embedding(tokens.clamp(max=self.vocab_size - 1))
        binary_emb = self.binary_proj(binary)
        return base + binary_emb
