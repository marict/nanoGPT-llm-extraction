#!/usr/bin/env python3
"""
Test script to verify that sample_dag_adjacency preserves ENTER -> EXIT paths
when given properly biased U_logits.
"""

import torch
import torch.nn.functional as F


def sample_dag_adjacency(U_logits, temperature=1.0, hard=True):
    """
    Sample a DAG adjacency matrix from logits using Gumbel softmax.

    Args:
        U_logits: (N, N) logits for edge probabilities
        temperature: Gumbel softmax temperature
        hard: Whether to use hard (discrete) or soft sampling

    Returns:
        adjacency: (N, N) adjacency matrix
    """
    # Apply softmax to get probabilities for each node's outgoing edges
    # We want each row to be a probability distribution over outgoing edges
    probs = F.softmax(U_logits, dim=1)

    # Sample using Gumbel softmax
    if hard:
        # For hard sampling, we need to be careful about the interpretation
        # Option 1: Sample each edge independently
        edge_samples = F.gumbel_softmax(U_logits, tau=temperature, hard=True, dim=1)
        return edge_samples
    else:
        # Soft sampling
        edge_samples = F.gumbel_softmax(U_logits, tau=temperature, hard=False, dim=1)
        return edge_samples


def create_biased_logits(N, bias_value=10.0, noise_std=1.0):
    """
    Create U_logits with strong bias for ENTER -> EXIT path.

    Args:
        N: Number of nodes
        bias_value: Large positive value to bias ENTER -> EXIT
        noise_std: Standard deviation for random noise

    Returns:
        U_logits: (N, N) biased logits
    """
    # Create random logits
    U_logits = torch.randn(N, N) * noise_std

    # Apply the bias for ENTER -> EXIT path
    U_mask = torch.ones_like(U_logits)
    U_mask[0, -1] = 0.0  # Zero out the random value at ENTER -> EXIT

    U_bias = torch.zeros_like(U_logits)
    U_bias[0, -1] = bias_value  # Add large positive bias

    U_logits = U_logits * U_mask + U_bias

    return U_logits


def test_enter_exit_path():
    """Test that ENTER -> EXIT path is preserved with high probability across 100 different DAGs."""
    print("=" * 60)
    print("Testing ENTER -> EXIT path preservation across 100 different DAGs")
    print("=" * 60)

    N = 5  # Small DAG for testing
    bias_value = 10.0
    num_different_dags = 100

    print(f"DAG size: {N} nodes")
    print(f"ENTER node: 0, EXIT node: {N-1}")
    print(f"Bias value: {bias_value}")
    print(f"Number of different DAGs to test: {num_different_dags}")
    print()

    # Test multiple sampling methods
    methods = [
        ("Hard Gumbel (temp=0.1)", {"temperature": 0.1, "hard": True}),
        ("Hard Gumbel (temp=1.0)", {"temperature": 1.0, "hard": True}),
        ("Soft Gumbel (temp=0.1)", {"temperature": 0.1, "hard": False}),
        ("Soft Gumbel (temp=1.0)", {"temperature": 1.0, "hard": False}),
    ]

    for method_name, kwargs in methods:
        print(f"Testing {method_name}:")

        success_count = 0
        enter_exit_values = []
        softmax_probs = []

        for dag_idx in range(num_different_dags):
            # Create different biased logits for each DAG
            U_logits = create_biased_logits(N, bias_value)

            # Record softmax probability for analysis
            probs = F.softmax(U_logits, dim=1)
            enter_exit_prob = probs[0, -1].item()
            softmax_probs.append(enter_exit_prob)

            # Sample adjacency matrix
            adjacency = sample_dag_adjacency(U_logits, **kwargs)

            # Check if ENTER -> EXIT edge exists
            enter_exit_value = adjacency[0, -1].item()
            enter_exit_values.append(enter_exit_value)

            # For hard sampling, check if edge is present (value > 0.5)
            # For soft sampling, just record the value
            if kwargs["hard"]:
                if enter_exit_value > 0.5:
                    success_count += 1
            else:
                if enter_exit_value > 0.1:  # Threshold for soft sampling
                    success_count += 1

        success_rate = success_count / num_different_dags
        avg_value = sum(enter_exit_values) / len(enter_exit_values)
        avg_softmax_prob = sum(softmax_probs) / len(softmax_probs)

        print(
            f"  Success rate: {success_rate:.2%} ({success_count}/{num_different_dags})"
        )
        print(f"  Average ENTER->EXIT sampled value: {avg_value:.4f}")
        print(f"  Average softmax probability: {avg_softmax_prob:.4f}")
        print(
            f"  Min/Max sampled values: {min(enter_exit_values):.4f} / {max(enter_exit_values):.4f}"
        )
        print(
            f"  Min/Max softmax probs: {min(softmax_probs):.4f} / {max(softmax_probs):.4f}"
        )

        # Check if success rate is acceptable
        expected_min_rate = 0.95 if kwargs["hard"] else 0.90
        if success_rate >= expected_min_rate:
            print(
                f"  ✅ PASS: Success rate {success_rate:.2%} >= {expected_min_rate:.2%}"
            )
        else:
            print(
                f"  ❌ FAIL: Success rate {success_rate:.2%} < {expected_min_rate:.2%}"
            )
        print()


def test_different_bias_values():
    """Test how different bias values affect success rate across 100 different DAGs each."""
    print("=" * 60)
    print("Testing different bias values across 100 different DAGs each")
    print("=" * 60)

    N = 4
    bias_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    num_different_dags = 100

    print(f"DAG size: {N} nodes")
    print(f"Different DAGs per bias value: {num_different_dags}")
    print()

    for bias_value in bias_values:
        success_count = 0
        softmax_probs = []

        for dag_idx in range(num_different_dags):
            # Create different biased logits for each DAG
            U_logits = create_biased_logits(N, bias_value)

            # Record softmax probability
            probs = F.softmax(U_logits, dim=1)
            enter_exit_prob = probs[0, -1].item()
            softmax_probs.append(enter_exit_prob)

            # Sample and test
            adjacency = sample_dag_adjacency(U_logits, temperature=1.0, hard=True)
            if adjacency[0, -1].item() > 0.5:
                success_count += 1

        success_rate = success_count / num_different_dags
        avg_softmax_prob = sum(softmax_probs) / len(softmax_probs)
        theoretical_prob = torch.exp(torch.tensor(bias_value)) / (
            torch.exp(torch.tensor(bias_value)) + (N - 1)
        )

        print(
            f"Bias {bias_value:4.1f}: Success rate {success_rate:.2%}, "
            f"Avg softmax prob {avg_softmax_prob:.2%}, "
            f"Theoretical prob {theoretical_prob:.2%}"
        )


def test_adjacency_properties():
    """Test other properties of the sampled adjacency matrix across different DAGs."""
    print("=" * 60)
    print("Testing adjacency matrix properties across different DAGs")
    print("=" * 60)

    N = 6
    num_samples = 10

    print(f"DAG size: {N} nodes")
    print(f"Number of samples to analyze: {num_samples}")
    print()

    total_edges_counts = []
    enter_edges_counts = []
    exit_edges_counts = []
    enter_exit_preserved = 0

    # Sample multiple adjacency matrices from different DAGs
    for i in range(num_samples):
        # Create different biased logits for each sample
        U_logits = create_biased_logits(N, bias_value=10.0)
        adjacency = sample_dag_adjacency(U_logits, temperature=1.0, hard=True)

        total_edges = (adjacency > 0.5).sum().item()
        enter_edges = (adjacency[0] > 0.5).sum().item()
        exit_edges = (adjacency[:, -1] > 0.5).sum().item()
        enter_exit_edge = adjacency[0, -1].item() > 0.5

        total_edges_counts.append(total_edges)
        enter_edges_counts.append(enter_edges)
        exit_edges_counts.append(exit_edges)
        if enter_exit_edge:
            enter_exit_preserved += 1

        if i < 3:  # Show detailed info for first 3 samples
            print(f"Sample {i+1}:")
            print(f"  Shape: {adjacency.shape}")
            print(f"  ENTER->EXIT edge: {adjacency[0, -1].item():.3f}")
            print(f"  Total edges: {total_edges}")
            print(f"  Edges from ENTER: {enter_edges}")
            print(f"  Edges to EXIT: {exit_edges}")

            # Print the adjacency matrix
            print("  Adjacency matrix:")
            for row in range(N):
                row_str = "    " + " ".join(
                    [f"{adjacency[row, col].item():.1f}" for col in range(N)]
                )
                print(row_str)
            print()

    # Print summary statistics
    print("Summary statistics across all samples:")
    print(
        f"  ENTER->EXIT preserved: {enter_exit_preserved}/{num_samples} ({100*enter_exit_preserved/num_samples:.1f}%)"
    )
    print(
        f"  Total edges - Mean: {sum(total_edges_counts)/len(total_edges_counts):.1f}, Range: {min(total_edges_counts)}-{max(total_edges_counts)}"
    )
    print(
        f"  Edges from ENTER - Mean: {sum(enter_edges_counts)/len(enter_edges_counts):.1f}, Range: {min(enter_edges_counts)}-{max(enter_edges_counts)}"
    )
    print(
        f"  Edges to EXIT - Mean: {sum(exit_edges_counts)/len(exit_edges_counts):.1f}, Range: {min(exit_edges_counts)}-{max(exit_edges_counts)}"
    )
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all tests
    test_enter_exit_path()
    test_different_bias_values()
    test_adjacency_properties()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
