import torch


def cast_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return *src* cast to the same dtype as *ref* (no-op if already aligned)."""
    return src if src.dtype == ref.dtype else src.to(ref.dtype)


def index_copy_like(
    dst: torch.Tensor,
    dim: int,
    idx: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    """Perform `index_copy` after automatically aligning *src* to *dst*'s dtype.

    If *src* is missing the dimension *dim* (common when updating a slice), it will
    be unsqueezed so its shape matches *dst* along that axis.
    """
    src = cast_like(src, dst)
    if src.dim() == dst.dim() - 1:
        src = src.unsqueeze(dim)
    return dst.index_copy(dim, idx, src)


__all__ = ["cast_like", "index_copy_like"]
