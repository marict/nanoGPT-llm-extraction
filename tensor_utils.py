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


def digits_to_magnitude(
    digits: torch.Tensor,
    max_digits: int,
    max_decimal_places: int,
    base: int = 10,
) -> torch.Tensor:
    """Convert a digit tensor to absolute magnitude.

    Args:
        digits: (..., D, base) tensor where the last dimension contains digit
            probabilities (or one-hot values) for each decimal place.
        max_digits: integer digits (D1).
        max_decimal_places: fractional digits (D2).
        base: number base for digit representation.

    Returns:
        magnitude: tensor with shape digits.shape[:-2]
    """
    device, dtype = digits.device, digits.dtype
    digits_vals = (digits * torch.arange(base, device=device, dtype=dtype)).sum(
        -1
    )  # (..., D)

    int_weights = base ** torch.arange(
        max_digits - 1, -1, -1, device=device, dtype=dtype
    )
    frac_weights = base ** torch.arange(
        -1, -max_decimal_places - 1, -1, device=device, dtype=dtype
    )
    weights = torch.cat((int_weights, frac_weights))  # (D,)
    magnitude = (digits_vals * weights).sum(-1)
    return magnitude


__all__ = ["cast_like", "index_copy_like", "digits_to_magnitude"]
