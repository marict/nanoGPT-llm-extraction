#!/usr/bin/env python
"""Example usage of the DAG dataset showing text expressions and their tensor labels."""

import random

import torch
from streaming import DAGStructureDataset


def print_example(text: str, structure: dict):
    """Pretty print a DAG example with its tensor labels."""
    print("\nText expression (from dataset):")
    print(f"  {text}")

    print("\nInitial values:")
    initial_values = []
    for i, (sgn, log) in enumerate(
        zip(structure["initial_sgn"], structure["initial_log"])
    ):
        val = float(sgn) * torch.exp(torch.tensor(float(log)))
        initial_values.append(val)
        print(
            f"  v{i}: sign={float(sgn):.1f}, log_mag={float(log):.3f} → {float(val):.3f}"
        )

    print("\nOperations (one-hot) and actual computation:")
    op_names = ["add", "subtract", "multiply", "divide", "identity"]
    values = initial_values.copy()

    for i, probs in enumerate(structure["operation_probs"]):
        op_idx = torch.argmax(probs).item()
        op_name = op_names[op_idx]

        # Get operand indices from the DAG plan
        if i < len(structure["operations"]):
            op1_idx, op2_idx, _ = structure["operations"][i]
            val1, val2 = values[op1_idx], values[op2_idx]

            # Compute result
            if op_name == "add":
                result = val1 + val2
                op_str = f"{val1:.3f} + {val2:.3f} = {result:.3f}"
            elif op_name == "subtract":
                result = val1 - val2
                op_str = f"{val1:.3f} - {val2:.3f} = {result:.3f}"
            elif op_name == "multiply":
                result = val1 * val2
                op_str = f"{val1:.3f} * {val2:.3f} = {result:.3f}"
            elif op_name == "divide":
                result = val1 / val2
                op_str = f"{val1:.3f} / {val2:.3f} = {result:.3f}"
            else:  # identity
                result = val1
                op_str = f"keep {val1:.3f}"

            values.append(result)

            print(f"  Step {i}: {op_name}")
            print(f"    Using v{op1_idx} and v{op2_idx}: {op_str}")
            probs_str = ", ".join(f"{float(p):.1f}" for p in probs)
            print(f"    Probabilities: [{probs_str}]")

    print("\nActual computation being performed:")
    if len(structure["operations"]) >= 2:
        op1_idx_0, op2_idx_0, _ = structure["operations"][0]
        op1_idx_1, op2_idx_1, _ = structure["operations"][1]
        op_idx_1 = torch.argmax(structure["operation_probs"][1]).item()
        op_name_1 = op_names[op_idx_1]

        if op_name_1 == "identity":
            # For identity operations, show just the first operation's result
            op_idx_0 = torch.argmax(structure["operation_probs"][0]).item()
            op_name_0 = op_names[op_idx_0]
            if op_name_0 == "add":
                print(
                    f"  {initial_values[op1_idx_0]:.3f} + {initial_values[op2_idx_0]:.3f}"
                )
            elif op_name_0 == "subtract":
                print(
                    f"  {initial_values[op1_idx_0]:.3f} - {initial_values[op2_idx_0]:.3f}"
                )
            elif op_name_0 == "multiply":
                print(
                    f"  {initial_values[op1_idx_0]:.3f} * {initial_values[op2_idx_0]:.3f}"
                )
            else:  # divide
                print(
                    f"  {initial_values[op1_idx_0]:.3f} / {initial_values[op2_idx_0]:.3f}"
                )
        else:
            # For non-identity operations, show the full computation
            if op1_idx_1 == 0:  # First operand is v0
                print(
                    f"  {initial_values[0]:.3f} * ({initial_values[op1_idx_0]:.3f} - {initial_values[op2_idx_0]:.3f})"
                )
            else:  # First operand is result of previous operation
                print(
                    f"  ({initial_values[op1_idx_0]:.3f} - {initial_values[op2_idx_0]:.3f}) * {initial_values[op2_idx_1]:.3f}"
                )
    else:
        print("  Simple computation with < 2 operations")


def main():
    # Create dataset with fixed seed for reproducibility
    dataset = DAGStructureDataset(max_depth=2, seed=42)

    # Generate a few examples
    print("Generating DAG examples with depth=2...")
    for i in range(3):
        print(f"\n=== Example {i+1} ===")
        text, structure = dataset.generate_structure_example(depth=2)
        print_example(text, structure)


if __name__ == "__main__":
    main()
