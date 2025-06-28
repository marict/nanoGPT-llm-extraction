import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from visualize_dag import visualize_dag


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
    Hard DAG adjacency sampling with guaranteed ENTER→EXIT path.
    """
    N = U_logits.size(0)
    P_hard = logits_to_hard_permutation_matrix(perm_logits)

    # Build U_hard with upper-triangular mask and threshold
    U = torch.sigmoid(U_logits) * torch.triu(torch.ones_like(U_logits), diagonal=1)
    U_hard = (U > edge_thresh).float()

    A_hard = P_hard.T @ U_hard @ P_hard
    A_hard.fill_diagonal_(0)
    return A_hard


def sample_ops_soft(op_logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    return F.gumbel_softmax(op_logits, tau=tau, hard=False, dim=-1)


def sample_ops_hard(op_logits: torch.Tensor) -> torch.Tensor:
    return F.gumbel_softmax(op_logits, tau=0.5, hard=True, dim=-1)


def enforce_enter_exit_constraints(A, enter_node=0, exit_node=-1):
    # Enter node: no incoming edges
    A[:, enter_node] = 0
    # Exit node: no outgoing edges
    A[exit_node, :] = 0
    A[enter_node, exit_node] = 1.0
    return A


ops = ["PLUS", "SUBTRACT", "MULTIPLY"]
K = len(ops)

N = 20

U_logits = torch.randn(N, N, requires_grad=True)
perm_logits = torch.randn(N, N, requires_grad=True)
op_logits = torch.randn(
    N - 2, K, requires_grad=True
)  # learnable or predicted by a model

A = enforce_enter_exit_constraints(sample_dag_adjacency(U_logits, perm_logits))
print(A)

A_hard = enforce_enter_exit_constraints(
    sample_dag_adjacency_hard(U_logits, perm_logits)
)
print(A_hard)

chosen = sample_ops_hard(op_logits)  # (N, K) one-hot
op_indices = chosen.argmax(dim=-1)  # (N,)
node_ops = ["ENTER"] + [ops[i] for i in op_indices] + ["EXIT"]

# visualize_dag(A_hard, node_labels=node_ops)

# Create a simple upper triangular matrix DAG
A = torch.randn(N, N)
A = A * torch.triu(torch.ones_like(A), diagonal=1)
A = enforce_enter_exit_constraints(A)
visualize_dag(A, node_labels=["ENTER"] + [i for i in range(N - 2)] + ["EXIT"])
