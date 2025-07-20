import argparse

from config import train_predictor_config as pred_config
from data.dagset.streaming import generate_single_dag_example

"""
Script to generate a random example from the DAG dataset.
Defaults to the config in config/train_predictor_config.py.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=pred_config.max_dag_depth)
    parser.add_argument(
        "--english-conversion-probability",
        type=float,
        default=pred_config.english_conversion_probability,
    )
    parser.add_argument(
        "--integer-no-decimal-probability",
        type=float,
        default=pred_config.integer_no_decimal_probability,
    )
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-digits", type=int, default=pred_config.max_digits)
    parser.add_argument(
        "--max-decimal-places", type=int, default=pred_config.max_decimal_places
    )
    parser.add_argument("--allowed-operations", type=str, default=None)
    parser.add_argument("--execute-sympy", type=bool, default=True)
    parser.add_argument("--printing-style", type=str, default="sstr")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    printing_style_probs = {
        "sstr": 0,
        "pretty": 0,
        "ascii": 0,
        "latex": 0,
    }
    expression_simplification_probability = 1.0 if args.simplify else 0.0
    expression_expansion_probability = 1.0 if args.expand else 0.0
    printing_style_probs[args.printing_style] = 1.0
    example = generate_single_dag_example(
        depth=args.depth,
        seed=args.seed,
        english_conversion_probability=args.english_conversion_probability,
        integer_no_decimal_probability=args.integer_no_decimal_probability,
        expression_simplification_probability=expression_simplification_probability,
        expression_expansion_probability=expression_expansion_probability,
        max_digits=args.max_digits,
        max_decimal_places=args.max_decimal_places,
        allowed_operations=args.allowed_operations,
        execute_sympy=args.execute_sympy,
        printing_style_probs=printing_style_probs,
    )
    print(example)
