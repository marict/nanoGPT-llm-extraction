#!/usr/bin/env python3
"""
Benchmark script for DAG dataset generation pipeline components.

This script measures the performance of different stages in the DAG dataset
generation to identify bottlenecks and optimization opportunities.
"""

import os
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sympy
import torch
from tiktoken import get_encoding

from data.dagset.generate_expression import (
    GENERATION_OPS,
    _apply_sympy_op,
    generate_expression,
    generate_uniform_digit_number,
    string_to_expression,
)
from data.dagset.preprocess_invalid_expression import preprocess_invalid_expression
from data.dagset.streaming import (
    expression_to_tensors,
    expressions_to_tensors,
    float_to_digit_onehot,
    normalize_expression,
)
from models.dag_model import DAGExecutor


class PipelineBenchmark:
    """Benchmark different components of the DAG pipeline"""

    def __init__(self, samples=100, warmup=10):
        self.samples = samples
        self.warmup = warmup
        self.tokenizer = get_encoding("gpt2")
        self.results = {}

    def time_function(self, func, *args, **kwargs):
        """Time a function with warmup and multiple samples"""
        # Warmup
        for _ in range(self.warmup):
            try:
                func(*args, **kwargs)
            except:
                pass

        # Actual timing
        times = []
        for _ in range(self.samples):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                # Skip failed runs but count them
                continue

        if not times:
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "samples": 0,
            }

        return {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "samples": len(times),
        }

    def benchmark_expression_generation(
        self, depth=4, max_digits=6, max_decimal_places=6
    ):
        """Benchmark expression generation"""
        print(f"🧪 Benchmarking expression generation (depth={depth})...")

        def generate_single():
            seed = torch.randint(0, 10000, (1,)).item()
            return generate_expression(
                depth=depth,
                seed=seed,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=self.tokenizer,
            )

        result = self.time_function(generate_single)
        self.results[f"expression_generation_d{depth}"] = result
        return result

    def benchmark_expression_generation_detailed(
        self, depth=4, max_digits=6, max_decimal_places=6
    ):
        """Benchmark expression generation with detailed breakdown"""
        print(f"🔬 Benchmarking detailed expression generation (depth={depth})...")

        import random

        # 1. Benchmark operation selection
        def select_operations():
            seed = torch.randint(0, 10000, (1,)).item()
            rng = random.Random(seed)
            ops_set_no_identity = [op for op in GENERATION_OPS if op != "identity"]
            weights = [i + 1 for i in range(depth)]
            num_ops = rng.choices(range(depth), weights=weights, k=1)[0]
            return [rng.choice(ops_set_no_identity) for _ in range(num_ops)]

        result = self.time_function(select_operations)
        self.results[f"expr_gen_ops_d{depth}"] = result

        # 2. Benchmark initial value generation
        def generate_initial_values():
            seed = torch.randint(0, 10000, (1,)).item()
            num_ops = depth // 2  # Approximate
            values = []
            for i in range(num_ops + 1):
                value = generate_uniform_digit_number(
                    seed=seed + i,
                    max_digits=max_digits,
                    max_decimal_places=max_decimal_places,
                    base=10,
                    allow_zero=False,
                    integer_no_decimal_probability=0.0,
                )
                values.append(value)
            return values

        result = self.time_function(generate_initial_values)
        self.results[f"expr_gen_values_d{depth}"] = result

        # 3. Benchmark SymPy expression construction
        def build_sympy_expression():
            seed = torch.randint(0, 10000, (1,)).item()
            rng = random.Random(seed)

            # Generate operations and values
            ops_set_no_identity = [op for op in GENERATION_OPS if op != "identity"]
            weights = [i + 1 for i in range(depth)]
            num_ops = rng.choices(range(depth), weights=weights, k=1)[0]
            sym_ops = [rng.choice(ops_set_no_identity) for _ in range(num_ops)]

            initial_values = []
            for i in range(num_ops + 1):
                value = generate_uniform_digit_number(
                    seed=seed + i,
                    max_digits=max_digits,
                    max_decimal_places=max_decimal_places,
                    base=10,
                    allow_zero=False,
                    integer_no_decimal_probability=0.0,
                )
                initial_values.append(value)

            # Build expression
            symbols = [sympy.Symbol(str(val)) for val in initial_values]
            nodes = symbols.copy()

            for op_name in reversed(sym_ops):
                if len(nodes) >= 2:
                    top = nodes.pop()
                    second = nodes.pop()
                    expr = _apply_sympy_op(op_name, second, top)
                    nodes.append(expr)

            return nodes[0] if nodes else symbols[0]

        result = self.time_function(build_sympy_expression)
        self.results[f"expr_gen_sympy_d{depth}"] = result

        # 4. Benchmark tokenization
        def tokenize_expression():
            # Use a representative expression
            expr = sympy.sympify("2*3 + (-1*5) - 7/8", evaluate=False)
            expr_text = str(expr)
            return self.tokenizer.encode(expr_text)

        result = self.time_function(tokenize_expression)
        self.results[f"expr_gen_tokenize_d{depth}"] = result

        # 5. Benchmark substring generation
        def generate_substrings():
            # Use a representative expression
            expr_text = "2*3 + (-1*5) - 7/8"
            tokens = self.tokenizer.encode(expr_text)
            substrings = []
            for i in range(len(tokens)):
                substring_tokens = tokens[: i + 1]
                substring = self.tokenizer.decode(substring_tokens)
                substrings.append(substring)
            return substrings

        result = self.time_function(generate_substrings)
        self.results[f"expr_gen_substrings_d{depth}"] = result

        # 6. Benchmark expression parsing
        def parse_expressions():
            test_substrings = ["2", "2*3", "2*3 +", "2*3 + (-1", "2*3 + (-1*5)"]
            expressions = []
            for substring in test_substrings:
                try:
                    expr = string_to_expression(substring.strip())
                    float(expr)  # Validate
                    expressions.append(expr)
                except:
                    expressions.append("not valid")
            return expressions

        result = self.time_function(parse_expressions)
        self.results[f"expr_gen_parsing_d{depth}"] = result

        # 7. Benchmark preprocessing
        def preprocess_invalid():
            invalid_substrings = ["2*3 +", "2*3 + (-1", "(2*3) / ("]
            processed = []
            for substring in invalid_substrings:
                preprocessed = preprocess_invalid_expression(substring)
                if preprocessed:
                    try:
                        expr = string_to_expression(preprocessed)
                        float(expr)
                        processed.append(expr)
                    except:
                        processed.append("not valid")
                else:
                    processed.append("not valid")
            return processed

        result = self.time_function(preprocess_invalid)
        self.results[f"expr_gen_preprocess_d{depth}"] = result

        # 8. Benchmark evaluable verification
        def verify_evaluable():
            # Test different types of expressions
            test_expressions = [
                sympy.sympify("2", evaluate=False),
                sympy.sympify("2*3", evaluate=False),
                sympy.sympify("2*3 + 5", evaluate=False),
                sympy.sympify("(2*3 + 5) / 7", evaluate=False),
                sympy.sympify("(-1*5) + 3*4", evaluate=False),
            ]
            results = []
            for expr in test_expressions:
                try:
                    result = float(expr)  # This is the verification step
                    results.append(result)
                except:
                    results.append("not evaluable")
            return results

        result = self.time_function(verify_evaluable)
        self.results[f"expr_gen_verify_d{depth}"] = result

    def benchmark_normalization(self, complexity="medium"):
        """Benchmark expression normalization"""
        print(f"🔄 Benchmarking normalization ({complexity})...")

        # Create test expressions of different complexity
        if complexity == "simple":
            test_expr = sympy.sympify("-1 * 5", evaluate=False)
        elif complexity == "medium":
            test_expr = sympy.sympify("2*3 + (-1*5)", evaluate=False)
        elif complexity == "complex":
            test_expr = sympy.sympify("1/(2*3) + (-1*5) - 7*8", evaluate=False)
        else:
            test_expr = sympy.sympify(
                "1/(1/(2*3)) + (-1*(-1*5)) - 7*8/(9-1)", evaluate=False
            )

        result = self.time_function(normalize_expression, test_expr)
        self.results[f"normalization_{complexity}"] = result
        return result

    def benchmark_tensor_conversion(self, depth=4, max_digits=6, max_decimal_places=6):
        """Benchmark tensor conversion"""
        print(f"📊 Benchmarking tensor conversion (depth={depth})...")

        # Generate a sample expression first
        expressions, _, _ = generate_expression(
            depth=depth,
            seed=42,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            tokenizer=self.tokenizer,
        )

        # Filter to valid expressions
        valid_exprs = [expr for expr in expressions if expr != "not valid"]
        if not valid_exprs:
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "samples": 0,
            }

        test_expr = valid_exprs[0]  # Use first valid expression

        def convert_single():
            return expressions_to_tensors(
                [test_expr],
                depth=depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

        result = self.time_function(convert_single)
        self.results[f"tensor_conversion_d{depth}"] = result
        return result

    def benchmark_digit_conversion(self, max_digits=6, max_decimal_places=6):
        """Benchmark float to digit conversion"""
        print(f"🔢 Benchmarking digit conversion...")

        # Test different value ranges
        test_values = [1.2345, 123.456, 0.001234, 999999.999999, -1234.5678]

        for i, value in enumerate(test_values):

            def convert_single():
                return float_to_digit_onehot(value, max_digits, max_decimal_places)

            result = self.time_function(convert_single)
            self.results[f"digit_conversion_val{i}"] = result

        return self.results

    def benchmark_dag_execution(self, depth=4, max_digits=6, max_decimal_places=6):
        """Benchmark DAG execution"""
        print(f"⚡ Benchmarking DAG execution (depth={depth})...")

        # Generate and convert an expression to tensors
        expressions, _, _ = generate_expression(
            depth=depth,
            seed=42,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            tokenizer=self.tokenizer,
        )

        valid_exprs = [expr for expr in expressions if expr != "not valid"]
        if not valid_exprs:
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "samples": 0,
            }

        target_tensors, valid_mask = expressions_to_tensors(
            [valid_exprs[0]],
            depth=depth,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
        )

        if not valid_mask[0]:
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "samples": 0,
            }

        # Prepare inputs
        target = target_tensors[0]
        digit_logits = target["target_digits"].unsqueeze(0).unsqueeze(0)
        V_sign = target["target_V_sign"].unsqueeze(0).unsqueeze(0)
        O = target["target_O"].unsqueeze(0).unsqueeze(0)
        G = target["target_G"].unsqueeze(0).unsqueeze(0)

        executor = DAGExecutor(
            dag_depth=depth,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
        )

        def execute_single():
            return executor.forward(digit_logits, V_sign, O, G)

        result = self.time_function(execute_single)
        self.results[f"dag_execution_d{depth}"] = result
        return result

    def benchmark_full_pipeline(self, depth=4, max_digits=6, max_decimal_places=6):
        """Benchmark the complete pipeline end-to-end"""
        print(f"🔄 Benchmarking full pipeline (depth={depth})...")

        def full_pipeline():
            # Generate expression
            seed = torch.randint(0, 10000, (1,)).item()
            expressions, _, _ = generate_expression(
                depth=depth,
                seed=seed,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=self.tokenizer,
            )

            # Filter valid expressions
            valid_exprs = [expr for expr in expressions if expr != "not valid"]
            if not valid_exprs:
                return None

            # Convert to tensors
            target_tensors, valid_mask = expressions_to_tensors(
                valid_exprs[:1],  # Just first expression
                depth=depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            if not valid_mask[0]:
                return None

            # Execute DAG
            target = target_tensors[0]
            digit_logits = target["target_digits"].unsqueeze(0).unsqueeze(0)
            V_sign = target["target_V_sign"].unsqueeze(0).unsqueeze(0)
            O = target["target_O"].unsqueeze(0).unsqueeze(0)
            G = target["target_G"].unsqueeze(0).unsqueeze(0)

            executor = DAGExecutor(
                dag_depth=depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            return executor.forward(digit_logits, V_sign, O, G)

        result = self.time_function(full_pipeline)
        self.results[f"full_pipeline_d{depth}"] = result
        return result

    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("🚀 Starting comprehensive DAG pipeline benchmark...\n")

        # Test different depths
        depths = [3, 4, 5, 6]

        for depth in depths:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING DEPTH {depth}")
            print(f"{'='*60}")

            # Benchmark each component
            self.benchmark_expression_generation(depth=depth)
            self.benchmark_expression_generation_detailed(depth=depth)
            self.benchmark_tensor_conversion(depth=depth)
            self.benchmark_dag_execution(depth=depth)
            self.benchmark_full_pipeline(depth=depth)

        # Benchmark other components (depth-independent)
        print(f"\n{'='*60}")
        print(f"BENCHMARKING OTHER COMPONENTS")
        print(f"{'='*60}")

        for complexity in ["simple", "medium", "complex", "very_complex"]:
            self.benchmark_normalization(complexity=complexity)

        self.benchmark_digit_conversion()

        print("\n✅ Benchmark complete!")

    def print_results(self):
        """Print benchmark results in a nice format"""
        print(f"\n{'='*80}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*80}")

        # Group results by category
        categories = {
            "Expression Generation (Overall)": [
                k for k in self.results.keys() if "expression_generation_d" in k
            ],
            "Expression Generation (Breakdown)": [
                k for k in self.results.keys() if "expr_gen_" in k
            ],
            "Normalization": [k for k in self.results.keys() if "normalization" in k],
            "Tensor Conversion": [
                k for k in self.results.keys() if "tensor_conversion" in k
            ],
            "DAG Execution": [k for k in self.results.keys() if "dag_execution" in k],
            "Digit Conversion": [
                k for k in self.results.keys() if "digit_conversion" in k
            ],
            "Full Pipeline": [k for k in self.results.keys() if "full_pipeline" in k],
        }

        for category, keys in categories.items():
            if not keys:
                continue

            print(f"\n📊 {category}")
            print("-" * 60)

            for key in sorted(keys):
                result = self.results[key]
                if result["samples"] == 0:
                    print(f"  {key:30s}: FAILED")
                else:
                    print(
                        f"  {key:30s}: {result['mean']*1000:8.3f}ms ± {result['std']*1000:6.3f}ms "
                        f"(min: {result['min']*1000:6.3f}ms, max: {result['max']*1000:6.3f}ms, n={result['samples']})"
                    )

        # Find bottlenecks
        print(f"\n🔍 BOTTLENECK ANALYSIS")
        print("-" * 60)

        # Sort by mean time
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if v["samples"] > 0],
            key=lambda x: x[1]["mean"],
            reverse=True,
        )

        print("Top 5 slowest operations:")
        for i, (name, result) in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {name:30s}: {result['mean']*1000:8.3f}ms")

        # Compare depth scaling
        print(f"\n📈 DEPTH SCALING ANALYSIS")
        print("-" * 60)

        for operation in ["full_pipeline", "tensor_conversion", "dag_execution"]:
            times_by_depth = {}
            for key, result in self.results.items():
                if operation in key and result["samples"] > 0:
                    depth = key.split("_d")[-1]
                    times_by_depth[depth] = result["mean"] * 1000

            if len(times_by_depth) > 1:
                print(f"\n{operation.replace('_', ' ').title()}:")
                for depth in sorted(times_by_depth.keys()):
                    print(f"  Depth {depth}: {times_by_depth[depth]:8.3f}ms")


def main():
    """Run the benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark DAG pipeline components")
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples per benchmark"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (fewer samples)"
    )

    args = parser.parse_args()

    if args.quick:
        args.samples = 20
        args.warmup = 3

    print(
        f"Running benchmark with {args.samples} samples and {args.warmup} warmup iterations..."
    )

    benchmark = PipelineBenchmark(samples=args.samples, warmup=args.warmup)
    benchmark.run_comprehensive_benchmark()
    benchmark.print_results()


if __name__ == "__main__":
    main()
