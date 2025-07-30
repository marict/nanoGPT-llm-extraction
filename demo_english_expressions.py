"""Demonstration of English expression training integration.

This shows how the expression_to_english module would integrate with
the existing DAG expression generation system to enable training on
English descriptions of mathematical expressions.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute()))

import tiktoken

from data.dagset.streaming import generate_expression
from expression_to_english import english_to_expression, expression_to_english


def generate_english_expression(
    *,
    depth: int,
    seed: int,
    max_digits: int,
    max_decimal_places: int,
    tokenizer,  # tiktoken.Encoding
    base: int = 10,
    english_probability: float = 0.5,  # Probability of using English instead of math notation
    _enable_preprocessing: bool = True,
) -> tuple[list[str], list[str], list[bool]]:
    """Generate expressions with optional English conversion.

    This is similar to generate_expression() but with an additional step
    to convert some expressions to English descriptions.

    Args:
        depth: Target depth for DAG operations
        seed: Random seed for reproducibility
        max_digits: Maximum number of digits for encoding
        max_decimal_places: Maximum decimal places for encoding
        tokenizer: tiktoken tokenizer for creating per-token substrings
        base: Number base (default 10)
        english_probability: Probability of converting to English (0.0 = never, 1.0 = always)
        _enable_preprocessing: Whether to apply preprocessing

    Returns:
        Tuple of (expressions, substrings, valid_mask) where:
        - expressions: List of sympy expressions or "not valid" strings
        - substrings: List of corresponding string representations (possibly English)
        - valid_mask: Boolean list indicating which expressions are valid
    """
    # Generate the base mathematical expression
    base_expressions, math_substrings, base_valid_mask = generate_expression(
        depth=depth,
        seed=seed,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        tokenizer=tokenizer,
        base=base,
        _enable_preprocessing=_enable_preprocessing,
    )

    # Decide whether to use English conversion
    import random

    rng = random.Random(seed + 12345)  # Different seed for English decision
    use_english = rng.random() < english_probability

    if not use_english:
        # Return original mathematical notation
        return base_expressions, math_substrings, base_valid_mask

    # Convert the final expression to English
    final_math_expression = math_substrings[-1] if math_substrings else ""
    english_expression = expression_to_english(final_math_expression)

    if not english_expression:
        # English conversion failed, fall back to math notation
        return base_expressions, math_substrings, base_valid_mask

    print(f"🔄 Converted: '{final_math_expression}' -> '{english_expression}'")

    # Tokenize the English version
    english_tokens = tokenizer.encode(english_expression)

    # Generate per-token substrings for the English version
    english_substrings = []
    english_expressions = []

    for i in range(len(english_tokens)):
        # Get substring up to token i+1
        substring_tokens = english_tokens[: i + 1]
        english_substring = tokenizer.decode(substring_tokens)
        english_substrings.append(english_substring)

        # Try to convert back to mathematical expression
        math_equivalent = english_to_expression(english_substring.strip())

        if math_equivalent is not None:
            # Successfully converted back to math - try to parse as sympy
            try:
                from data.dagset.streaming import string_to_expression

                expr = string_to_expression(math_equivalent)
                # Verify it's evaluable
                float(expr)
                english_expressions.append(expr)
            except:
                english_expressions.append("not valid")
        else:
            english_expressions.append("not valid")

    # Create validity mask
    english_valid_mask = [expr != "not valid" for expr in english_expressions]

    return english_expressions, english_substrings, english_valid_mask


def demo_english_expressions():
    """Demonstrate English expression generation."""
    print("🧮 Demo: English Expression Generation")
    print("=" * 60)

    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate examples with different English probabilities
    for i, english_prob in enumerate([0.0, 1.0]):
        print(f"\n📊 Example {i+1}: english_probability = {english_prob}")
        print("-" * 40)

        expressions, substrings, valid_mask = generate_english_expression(
            depth=4,
            seed=42 + i,
            max_digits=4,
            max_decimal_places=4,
            tokenizer=tokenizer,
            english_probability=english_prob,
        )

        print(f'Final expression: "{substrings[-1]}"')
        print(f"Total tokens: {len(valid_mask)}")
        print(f"Valid tokens: {sum(valid_mask)}")
        print(f"Valid rate: {sum(valid_mask)/len(valid_mask):.1%}")

        # Show first few tokens
        print("\nPer-token breakdown (first 8 tokens):")
        for j in range(min(8, len(substrings))):
            status = "✅ VALID" if valid_mask[j] else "❌ INVALID"
            print(f'  {j+1:2d}: "{substrings[j][:30]:<30}" → {status}')

        if len(substrings) > 8:
            print(f"  ... and {len(substrings) - 8} more tokens")


def demo_conversion_quality():
    """Test conversion quality on various expressions."""
    print("\n🔍 Testing Conversion Quality")
    print("=" * 60)

    test_expressions = [
        "3.5 + 4.2",
        "42 / 7",
        "-5.3 + 85.0",
        "3.5 * (1.8 - 6.7)",
        "(3.5 + 1.2) * 4.0",
    ]

    for expr in test_expressions:
        english = expression_to_english(expr)
        back = english_to_expression(english)

        # Normalize for comparison
        from expression_to_english import normalize_expression

        original_norm = normalize_expression(expr)
        back_norm = normalize_expression(back) if back else None
        reversible = back_norm == original_norm

        print(f"\nOriginal:  {expr}")
        print(f"English:   {english}")
        print(f"Back:      {back}")
        print(f"✅ Reversible: {reversible}")


if __name__ == "__main__":
    demo_english_expressions()
    demo_conversion_quality()
