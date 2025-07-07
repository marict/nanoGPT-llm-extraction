#!/usr/bin/env python
"""
Test script for mixed English/numeric conversion functionality.
Demonstrates per-token conversion that creates mixed expressions.
"""

import random

from data.dagset.streaming import (add_english_to_expression,
                                   convert_dag_text_to_english,
                                   generate_single_dag_example)


def test_mixed_conversion():
    """Test the new per-token English conversion functionality."""

    print("=== Mixed English/Numeric Conversion Test ===\n")

    # Create some test expressions
    test_expressions = [
        "19.4 / 85.97 / 8.3213",
        "(5.22 - 3.213) * 2.32",
        "77.101 * 45.5 + 14.7",
        "93.3 * 4.9 - 93.3",
        "12.5 + 6.8 / 3.2 - 1.1",
    ]

    print("Testing different conversion probabilities:")
    print("=" * 50)

    rng = random.Random(42)

    for prob in [0.2, 0.4, 0.6]:
        print(f"\nConversion probability: {prob}")
        print("-" * 30)

        for expr in test_expressions[:3]:  # Test first 3 expressions
            converted = add_english_to_expression(expr, prob, rng)
            print(f"  Original: {expr}")
            print(f"  Mixed:    {converted}")
            print()

    print("\n" + "=" * 60)
    print("Multiple conversions of same expression (showing randomness):")
    print("=" * 60)

    test_expr = "25.5 * 12.3 + 8.7 / 4.1"
    print(f"\nOriginal: {test_expr}")
    print("Multiple mixed conversions:")

    for i in range(8):
        converted = add_english_to_expression(test_expr, 0.5, rng)
        print(f"  {i+1}: {converted}")

    print("\n" + "=" * 60)
    print("Testing with generated DAG expressions:")
    print("=" * 60)

    for i in range(5):
        # Generate a DAG example
        depth = 2 + i % 3  # Depths 2, 3, 4
        dag_example = generate_single_dag_example(
            depth=depth, value_range=(1.0, 100.0), rng=random.Random(100 + i)
        )

        # Apply mixed conversion
        mixed_text = convert_dag_text_to_english(
            dag_example.text, conversion_probability=0.4, rng=random.Random(200 + i)
        )

        print(f"\nExample {i+1} (depth {depth}):")
        print(f"  Original: {dag_example.text}")
        print(f"  Mixed:    {mixed_text}")


if __name__ == "__main__":
    test_mixed_conversion()
