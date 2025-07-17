#!/usr/bin/env python3
"""
SoftGCD: differentiable approximation of gcd(n1, n2).

Pipeline:
1.  Detect primes dividing n via smooth Gaussian divisibility mask.
2.  For each dividing prime p, find soft exponent k ≈ v_p(n) with
    a differentiable max‑over‑candidates using torchsort.soft_sort.
3.  Form exponent vectors for n1 and n2, take soft element‑wise min,
    and reconstruct gcd̂.

A quick demo at the end verifies that gradients reach the inputs.
"""

from __future__ import annotations

import argparse
from typing import List

import torch
import torchsort
from sympy import primerange


# ---------------------------------------------------------------------------#
def generate_primes(limit: int) -> List[int]:
    """Return all primes ≤ limit using sympy.primerange()."""
    return list(primerange(0, limit + 1))


# ---------------------------------------------------------------------------#
class SoftGCD(torch.nn.Module):
    """Differentiable greatest‑common‑divisor approximation."""

    def __init__(
        self,
        primes: List[int],
        alpha: float = 200.0,
        beta: float = 2560.0,
        mask_power: int = 8,
        tau: float = 1e-2,
    ) -> None:
        """
        Args:
            primes: sorted list of primes used for factor search.
            alpha: sharpness of prime‑divisibility Gaussian.
            beta:  sharpness of exponent divisibility Gaussian.
            mask_power: exponent applied to prime mask (makes 0/1 crisper).
            tau:  regularization_strength for torchsort.soft_sort (smaller → harder).
        """
        super().__init__()
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.float32))
        self.register_buffer("log_primes", torch.log(self.primes))
        self.alpha = alpha
        self.beta = beta
        self.mask_power = mask_power
        self.tau = tau

    # ---------------------------------------------------------------------#
    def _prime_divisibility_mask(self, n: torch.Tensor) -> torch.Tensor:
        """Soft 0/1 mask indicating which primes divide n."""
        ratios = n / self.primes
        frac = ratios - torch.round(ratios)
        mask = torch.exp(-self.alpha * frac * frac)
        mask = mask**self.mask_power  # sharpen
        # safe divide trick to push exact zeros to zero
        mask = mask * (ratios / (ratios + (ratios == 0).float()))
        return mask  # shape (num_primes,)

    # ---------------------------------------------------------------------#
    def _soft_exponent(self, n: torch.Tensor) -> torch.Tensor:
        """Return soft exponent vector v_p(n) for all stored primes."""
        mask_p = self._prime_divisibility_mask(n)
        exponents = []
        for idx, p in enumerate(self.primes):
            if mask_p[idx] < 0.99:
                exponents.append(torch.tensor(0.0, device=n.device))
                continue
            # integer upper bound for k: floor(log_p n)
            max_k = int(torch.floor(torch.log(n) / self.log_primes[idx]).item())
            if max_k == 0:
                exponents.append(torch.tensor(0.0, device=n.device))
                continue
            k_candidates = torch.arange(
                1, max_k + 1, device=n.device, dtype=torch.float32
            )
            ratios = n / (p**k_candidates)
            frac = ratios - torch.round(ratios)
            mask_k = torch.exp(-self.beta * frac * frac)
            scores = mask_k * k_candidates  # (K,)
            # differentiable soft max via torchsort: take last element after sorting ascending
            sorted_scores = torchsort.soft_sort(
                scores[None, :],
                regularization_strength=self.tau,
            )  # shape (1, K)
            k_hat = sorted_scores[0, -1]
            exponents.append(k_hat)
        return torch.stack(exponents)  # shape (num_primes,)

    # ---------------------------------------------------------------------#
    def forward(self, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        """Compute differentiable gcd(n1, n2)."""
        e1 = self._soft_exponent(n1)
        e2 = self._soft_exponent(n2)
        stacked = torch.stack((e1, e2), dim=-1)  # (num_primes, 2)
        # differentiable soft-min via torchsort (first column)
        sorted_pair = torchsort.soft_sort(
            stacked,
            regularization_strength=self.tau,
        )  # shape (num_primes, 2)
        min_exp = sorted_pair[:, 0]
        log_gcd = (min_exp * self.log_primes).sum()
        return torch.exp(log_gcd)


# ---------------------------------------------------------------------------#
def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SoftGCD demo")
    parser.add_argument(
        "--n1", type=float, default=83160.0, help="first integer (≤ max-input)"
    )
    parser.add_argument(
        "--n2", type=float, default=123120.0, help="second integer (≤ max-input)"
    )
    parser.add_argument(
        "--max-input",
        type=float,
        default=1e6,
        help="maximum allowed value for n1 or n2 (controls prime search cost)",
    )
    return parser


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    args = parse_args().parse_args()

    # Validate input magnitude to keep the prime search tractable.
    max_val = max(args.n1, args.n2)
    if max_val > args.max_input:
        raise ValueError(
            f"Inputs too large (max={max_val}); increase --max-input if intentional."
        )

    primes = generate_primes(int(max_val))
    model = SoftGCD(primes)

    n1 = torch.tensor(args.n1, requires_grad=True)
    n2 = torch.tensor(args.n2, requires_grad=True)

    gcd_hat = model(n1, n2)
    print(f"Soft gcd̂ ≈ {gcd_hat.item():.3f}")

    # Simple gradient check
    loss = (gcd_hat - 1.0) ** 2
    loss.backward()
    print(f"∂loss/∂n1 = {n1.grad.item():.3e}")
    print(f"∂loss/∂n2 = {n2.grad.item():.3e}")
