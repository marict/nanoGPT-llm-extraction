"""
Test to verify that expression generation is fully deterministic with no random leakage.
"""

import pytest
from tiktoken import get_encoding

from data.dagset.generate_expression import generate_expression


def test_expression_generation_determinism():
    """
    Test that generating expressions with the same seed produces identical results.
    This verifies there's no random leakage in the generation process.
    """
    tokenizer = get_encoding("gpt2")

    # Test parameters
    depth = 4
    max_digits = 3
    max_decimal_places = 3
    num_generations = 10  # Generate 10 times with same seed

    # Test multiple different seeds to be thorough
    test_seeds = [42, 123, 999, 1337, 0]

    for seed in test_seeds:
        print(f"\n=== Testing seed {seed} ===")

        # Generate expressions multiple times with the same seed
        all_results = []
        for generation_round in range(num_generations):
            expressions, substrings, valid_mask = generate_expression(
                depth=depth,
                seed=seed,  # Same seed every time
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=tokenizer,
            )

            # Store results for comparison
            all_results.append(
                {
                    "expressions": [str(expr) for expr in expressions],
                    "substrings": substrings.copy(),
                    "valid_mask": valid_mask.copy(),
                    "count": len(expressions),
                }
            )

        # Verify all generations produced identical results
        reference = all_results[0]
        print(f"  Generated {reference['count']} expressions")

        for round_idx, result in enumerate(all_results[1:], 1):
            # Check expression count
            assert (
                result["count"] == reference["count"]
            ), f"Round {round_idx}: Expression count mismatch - expected {reference['count']}, got {result['count']}"

            # Check expressions
            assert (
                result["expressions"] == reference["expressions"]
            ), f"Round {round_idx}: Expression mismatch at seed {seed}"

            # Check substrings
            assert (
                result["substrings"] == reference["substrings"]
            ), f"Round {round_idx}: Substring mismatch at seed {seed}"

            # Check valid mask
            assert (
                result["valid_mask"] == reference["valid_mask"]
            ), f"Round {round_idx}: Valid mask mismatch at seed {seed}"

        print(f"  ✅ All {num_generations} generations identical")
        print(f"  Sample expressions: {reference['expressions'][:3]}")

    print(
        f"\n✅ DETERMINISM VERIFIED: All {len(test_seeds)} seeds produced consistent results across {num_generations} generations"
    )


def test_different_seeds_produce_different_results():
    """
    Test that different seeds produce different expressions (sanity check).
    """
    tokenizer = get_encoding("gpt2")

    # Test parameters
    depth = 4
    max_digits = 3
    max_decimal_places = 3

    # Generate with different seeds
    seeds = [42, 43, 44, 45, 46]
    all_results = []

    for seed in seeds:
        expressions, substrings, valid_mask = generate_expression(
            depth=depth,
            seed=seed,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            tokenizer=tokenizer,
        )

        all_results.append(
            {
                "seed": seed,
                "expressions": [str(expr) for expr in expressions],
                "count": len(expressions),
            }
        )

    # Verify that different seeds produce different results
    print(f"\n=== Testing different seeds produce different results ===")
    reference = all_results[0]
    different_count = 0

    for result in all_results[1:]:
        if result["expressions"] != reference["expressions"]:
            different_count += 1
            print(
                f"  Seed {result['seed']}: Different from seed {reference['seed']} ✅"
            )
        else:
            print(f"  Seed {result['seed']}: Same as seed {reference['seed']} ⚠️")

    # At least some seeds should produce different results
    assert (
        different_count > 0
    ), "All different seeds produced identical results - this suggests the seed isn't working"

    print(
        f"\n✅ RANDOMNESS VERIFIED: {different_count}/{len(seeds)-1} different seeds produced different results"
    )


if __name__ == "__main__":
    test_expression_generation_determinism()
    test_different_seeds_produce_different_results()
