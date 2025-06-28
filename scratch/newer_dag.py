import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from visualize_dag import visualize_dag


def enforce_enter_exit_constraints(A, enter_node=0, exit_node=-1):
    # Enter node: no incoming edges
    A[:, enter_node] = 0
    # Exit node: no outgoing edges
    A[exit_node, :] = 0
    A[enter_node, exit_node] = 1.0
    return A


ops = ["PLUS", "SUBTRACT", "MULTIPLY"]
K = len(ops)

N = 5
# Create a simple upper triangular matrix DAG
A = torch.sigmoid(torch.randn(N, N, requires_grad=True))
A = A * torch.triu(torch.ones_like(A), diagonal=1)
A = enforce_enter_exit_constraints(A)
visualize_dag(A, node_labels=["ENTER"] + [i for i in range(N - 2)] + ["EXIT"])
