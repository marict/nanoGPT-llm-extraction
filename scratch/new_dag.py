import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def gumbel_noise(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def sinkhorn(
    log_alpha: torch.Tensor, n_iters: int = 10, tau: float = 0.5
) -> torch.Tensor:
    """
    Applies Sinkhorn normalization to turn logits into a doubly stochastic matrix.
    """
    log_alpha = log_alpha / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha, dim=1, keepdim=True
        )  # row norm
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha, dim=0, keepdim=True
        )  # col norm
    return torch.exp(log_alpha)


def sample_permutation_matrix(
    logits: torch.Tensor, tau=0.5, n_iters=10
) -> torch.Tensor:
    gumbels = gumbel_noise(logits.shape)
    noisy_logits = logits + gumbels
    return sinkhorn(noisy_logits, n_iters=n_iters, tau=tau)


def sample_dag_adjacency(
    U_logits: torch.Tensor, perm_logits: torch.Tensor, tau=0.5
) -> torch.Tensor:
    N = U_logits.size(0)
    U = torch.sigmoid(U_logits) * torch.triu(torch.ones_like(U_logits), diagonal=1)
    P = sample_permutation_matrix(perm_logits, tau)
    print(f"P row sum: {P.sum(dim=1)}")
    print(f"P col sum: {P.sum(dim=0)}")
    A = P.T @ U @ P
    return A


def logits_to_hard_permutation_matrix(logits: torch.Tensor) -> torch.Tensor:
    cost = -logits.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    P = torch.zeros_like(logits)
    P[row_ind, col_ind] = 1.0
    return P.to(logits.device)


def sample_dag_adjacency_hard(
    U_logits: torch.Tensor, perm_logits: torch.Tensor, edge_thresh: float = 0.5
) -> torch.Tensor:
    """
    Hard version of DAG adjacency sampling for inference.
    - U_logits: logits for upper triangular matrix (N, N)
    - perm_logits: logits for node ordering (N, N)
    - edge_thresh: threshold for binarizing edge presence
    """
    N = U_logits.size(0)

    # Create hard permutation matrix (P_hard) using row-wise argmax
    P_hard = logits_to_hard_permutation_matrix(perm_logits)

    # Force upper-triangular structure in U, then threshold
    U = torch.sigmoid(U_logits) * torch.triu(torch.ones_like(U_logits), diagonal=1)
    U_hard = (U > edge_thresh).float()

    # Construct hard adjacency matrix A_hard = P^T U P
    A_hard = P_hard.T @ U_hard @ P_hard
    A_hard.fill_diagonal_(0)  # remove any self loops

    return A_hard


N = 5
U_logits = torch.randn(N, N, requires_grad=True)
perm_logits = torch.randn(N, N, requires_grad=True)
A = sample_dag_adjacency(U_logits, perm_logits)
print(A)

A_hard = sample_dag_adjacency_hard(U_logits, perm_logits)
print(A_hard)
