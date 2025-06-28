import matplotlib.pyplot as plt
import networkx as nx
import torch


def visualize_dag(A: torch.Tensor, node_labels=None, title="DAG Visualization"):
    """
    Visualizes a DAG given a (hard) adjacency matrix A (torch.Tensor of shape [N, N])
    """
    A_np = A.detach().cpu().numpy()
    G = nx.DiGraph()

    N = A_np.shape[0]
    labels = node_labels or [f"{i}" for i in range(N)]

    # Add nodes
    for i in range(N):
        G.add_node(i, label=labels[i])

    # Add edges
    for i in range(N):
        for j in range(N):
            if A_np[i, j] > 0:
                G.add_edge(i, j)

    # Draw
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={i: labels[i] for i in range(N)},
        node_color="skyblue",
        node_size=1500,
        font_size=12,
        arrows=True,
    )
    plt.title(title)
    plt.show()
