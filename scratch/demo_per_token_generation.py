#!/usr/bin/env python3
"""
Demo script showing per-token DAG generation and sub-expression analysis.

This script demonstrates how the system generates expressions and creates
per-token sub-expressions for dense training on all valid subexpressions.
"""

import argparse
import random
import sys

sys.path.insert(0, ".")

import tiktoken

from data.dagset.generate_expression import generate_expressions
from data.dagset.streaming import expressions_to_tensors


def demo_expression_generation(depth=6, seed=None):
    """Generate and analyze a complex expression."""

    # Generate random seed if none provided
    if seed is None:
        seed = random.randint(1, 10000)

    print(f"🚀 DEMO: Per-Token DAG Generation (depth={depth}, seed={seed})")
    print("=" * 65)

    # Get tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate expression
    expressions, substrings, valid_mask = generate_expressions(
        depth=depth,
        seed=seed,
        max_digits=3,
        max_decimal_places=4,
        tokenizer=tokenizer,
        _enable_preprocessing=True,
    )

    full_expr = substrings[-1]

    print(f"📝 GENERATED EXPRESSION:")
    print(f'   "{full_expr}"')
    print(f"📏 Length: {len(full_expr)} chars, {len(substrings)} tokens")
    print(
        f"📊 Valid: {sum(valid_mask)}/{len(valid_mask)} ({sum(valid_mask)/len(valid_mask):.1%})"
    )
    print()

    # Show token breakdown
    tokens = tokenizer.encode_ordinary(full_expr)
    token_strings = [
        tokenizer.decode_single_token_bytes(token).decode("utf-8", errors="replace")
        for token in tokens
    ]

    print("🔤 TOKEN BREAKDOWN:")
    print("-" * 40)
    for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
        repr_str = repr(token_str)[1:-1]  # Remove quotes, keep escape sequences
        print(f"  {i+1:2d}: {repr_str:>8} (id: {token_id})")
    print()

    # Show progression of sub-expressions
    print("🎭 SUB-EXPRESSION PROGRESSION:")
    print("-" * 65)

    for i, (expr, substring, is_valid) in enumerate(
        zip(expressions, substrings, valid_mask)
    ):
        pos = i + 1
        status = "✅" if is_valid else "❌"

        # Truncate long substrings for display
        display_sub = substring if len(substring) <= 35 else substring[:32] + "..."

        print(f'{pos:2d}: {status} "{display_sub}"')

        if is_valid:
            if hasattr(expr, "__class__"):
                expr_type = type(expr).__name__
                if hasattr(expr, "args") and len(expr.args) > 1:
                    expr_type += f" ({len(expr.args)} args)"
                print(f"    → {expr_type}: {expr}")
            else:
                print(f"    → {expr}")
        else:
            print(f"    → Invalid (masked during training)")
        print()

    # Report on invalid tokens specifically
    invalid_tokens = [
        (i + 1, substrings[i]) for i, valid in enumerate(valid_mask) if not valid
    ]

    print("🚨 INVALID TOKEN ANALYSIS:")
    print("-" * 40)
    if invalid_tokens:
        print(f"Found {len(invalid_tokens)} invalid tokens:")
        for pos, substring in invalid_tokens:
            # Analyze why it's invalid
            reasons = []
            open_parens = substring.count("(")
            close_parens = substring.count(")")

            if open_parens != close_parens:
                reasons.append(
                    f"unbalanced parens ({open_parens} open, {close_parens} close)"
                )

            if substring.endswith(("+", "-", "*", "/")):
                reasons.append("trailing operator")

            if substring.endswith("."):
                reasons.append("incomplete decimal")

            if substring.startswith(("+", "*", "/")):
                reasons.append("invalid leading operator")

            if "--" in substring:
                reasons.append("double negative")

            if not any(c.isdigit() for c in substring):
                reasons.append("no digits")

            # Truncate long substrings for display
            display_sub = substring if len(substring) <= 30 else substring[:27] + "..."
            reason_str = ", ".join(reasons) if reasons else "parsing failed"

            print(f'  {pos:2d}: "{display_sub}" → {reason_str}')
    else:
        print("✅ No invalid tokens - all substrings are valid!")
    print()

    # Show target tensor generation
    print("🎯 TARGET DAG TENSOR GENERATION:")
    print("-" * 40)

    try:
        target_tensors, tensor_valid_mask = expressions_to_tensors(
            expressions, depth=depth
        )

        print(f"✅ Generated {len(target_tensors)} target tensor dictionaries")
        print(
            f"✅ Tensor valid mask: {sum(tensor_valid_mask)}/{len(tensor_valid_mask)}"
        )

        # Show a few examples
        valid_positions = [i for i, valid in enumerate(tensor_valid_mask) if valid]

        print(f"\n📊 SAMPLE TARGET TENSORS (first 3 of {len(valid_positions)}):")
        for i, pos in enumerate(valid_positions[:3]):
            target_dict = target_tensors[pos]
            substring = substrings[pos]

            print(f'\nPosition {pos+1}: "{substring}"')
            print(f'  Initial values: {target_dict["target_initial_values"].tolist()}')

            # Get operation name
            op_probs = target_dict["target_operation_probs"]
            if op_probs.numel() > 0:
                op_idx = op_probs.argmax(dim=-1).tolist()[0]
                op_name = ["add", "multiply", "identity"][op_idx]
                print(f"  Primary operation: {op_name}")

            print(f'  Final execution: {target_dict["target_final_exec"]:.6f}')

    except Exception as e:
        print(f"❌ Error generating target tensors: {e}")

    print()
    print("🏆 KEY TAKEAWAYS:")
    print("-" * 20)
    print("✅ Each token position represents a meaningful sub-expression")
    print("✅ Invalid positions (incomplete syntax) are masked out")
    print("✅ Preprocessing converts many invalid strings to valid ones")
    print("✅ Model trains densely on ALL valid subexpressions")
    print("✅ This provides much richer training signal than final-token-only")
    print("✅ Each valid position gets independent DAG target tensors")

    return full_expr, len(valid_positions), len(substrings)


def compare_different_complexities():
    """Compare expressions of different depths."""

    print("\n" + "=" * 65)
    print("📈 COMPARING DIFFERENT EXPRESSION COMPLEXITIES")
    print("=" * 65)

    for depth in [2, 4, 6]:
        print(f"\n🎯 DEPTH {depth}:")
        expr, valid_count, total_count = demo_expression_generation(
            depth=depth, seed=789 + depth
        )
        print(f"   Result: {valid_count} valid tokens out of {total_count} total")
        print(f'   Expression: "{expr}"')


def run_multiple_demos(count=3, depth=6, seed=None):
    """Run the demo multiple times and summarize invalid token patterns."""

    print(f"🔄 RUNNING {count} DEMOS TO ANALYZE INVALID TOKEN PATTERNS")
    print("=" * 80)

    all_invalid_reasons = {}
    total_invalid = 0
    total_tokens = 0

    for run in range(count):
        print(f"\n🎲 RUN {run + 1}:")
        print("-" * 50)

        # Use provided seed or generate random one
        run_seed = seed if seed is not None else random.randint(1, 10000)

        # Get tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Generate expression
        expressions, substrings, valid_mask = generate_expressions(
            depth=depth,
            seed=run_seed,
            max_digits=3,
            max_decimal_places=4,
            tokenizer=tokenizer,
            _enable_preprocessing=True,
        )

        print(f"Seed: {run_seed}")
        print(f'Expression: "{substrings[-1]}"')
        print(
            f"Valid tokens: {sum(valid_mask)}/{len(valid_mask)} ({sum(valid_mask)/len(valid_mask):.1%})"
        )

        # Analyze invalid tokens
        invalid_count = len(valid_mask) - sum(valid_mask)
        total_invalid += invalid_count
        total_tokens += len(valid_mask)

        if invalid_count > 0:
            print(f"Invalid tokens:")
            for i, (valid, substring) in enumerate(zip(valid_mask, substrings)):
                if not valid:
                    reasons = []
                    open_parens = substring.count("(")
                    close_parens = substring.count(")")

                    if open_parens != close_parens:
                        reason = f"unbalanced parens ({open_parens}→{close_parens})"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if substring.endswith(("+", "-", "*", "/")):
                        reason = "trailing operator"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if substring.endswith("."):
                        reason = "incomplete decimal"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if substring.startswith(("+", "*", "/")):
                        reason = "invalid leading operator"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if "--" in substring:
                        reason = "double negative"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if not any(c.isdigit() for c in substring):
                        reason = "no digits"
                        reasons.append(reason)
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )

                    if not reasons:
                        reason = "parsing failed"
                        all_invalid_reasons[reason] = (
                            all_invalid_reasons.get(reason, 0) + 1
                        )
                        reasons = [reason]

                    display_sub = (
                        substring if len(substring) <= 25 else substring[:22] + "..."
                    )
                    print(f'  Token {i+1}: "{display_sub}" → {", ".join(reasons)}')
        else:
            print("✅ All tokens valid!")

    # Summary
    print(f"\n📊 SUMMARY ACROSS {count} RUNS:")
    print("=" * 50)
    print(f"Total tokens: {total_tokens}")
    print(f"Invalid tokens: {total_invalid} ({total_invalid/total_tokens:.1%})")
    print(
        f"Valid tokens: {total_tokens - total_invalid} ({(total_tokens - total_invalid)/total_tokens:.1%})"
    )

    if all_invalid_reasons:
        print(f"\nInvalid token reasons (frequency):")
        for reason, count in sorted(
            all_invalid_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {reason}: {count} occurrences")
    else:
        print("\n✅ No invalid tokens found across all runs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo per-token DAG generation")
    parser.add_argument(
        "--seed", type=int, help="Random seed (if not provided, uses random)"
    )
    parser.add_argument(
        "--depth", type=int, default=6, help="Expression depth (default: 6)"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs (default: 1)"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Run multiple demos and analyze patterns",
    )

    args = parser.parse_args()

    if args.multiple or args.runs > 1:
        run_multiple_demos(count=args.runs, depth=args.depth, seed=args.seed)
    else:
        demo_expression_generation(depth=args.depth, seed=args.seed)

        print("\n" + "=" * 65)
        print("✨ Demo complete! Run this script anytime to see examples.")
        print("✨ Try --multiple or --runs 3 to analyze invalid token patterns.")
        print("=" * 65)
