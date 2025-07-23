import argparse
import json
from pathlib import Path

import torch

from evaluation import (
    comprehensive_evaluate,
    evaluate_from_dataset_file,
    evaluate_math,
    load_checkpoint,
)
from python_version_check import check_python_version

check_python_version()


def compare_models_comprehensive(
    ckpt_baseline: str,
    ckpt_dag: str,
    data_dir: str = None,
    dataset_file: str = None,
    eval_iters: int = 200,
    batch_size: int = 8,
    math_tasks: list = None,
    math_max_examples: int = 50,
    save_results: str = None,
):
    """
    Compare two models using all evaluation metrics from the training loop.

    Args:
        ckpt_baseline: Path to baseline checkpoint
        ckpt_dag: Path to DAG checkpoint
        data_dir: Directory with train.bin/val.bin for comprehensive evaluation
        dataset_file: Simple text file for basic loss evaluation (fallback)
        eval_iters: Number of iterations for loss estimation
        batch_size: Batch size for evaluation
        math_tasks: Math evaluation tasks
        math_max_examples: Max examples per math task
        save_results: Optional path to save results as JSON
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if math_tasks is None:
        math_tasks = ["gsm8k", "svamp"]

    # Load models
    print(f"Loading baseline model from {ckpt_baseline}")
    baseline_model, baseline_ckpt = load_checkpoint(ckpt_baseline, device)

    print(f"Loading DAG model from {ckpt_dag}")
    dag_model, dag_ckpt = load_checkpoint(ckpt_dag, device)

    results = {"baseline": {}, "dag": {}, "comparison": {}}

    # Add model info
    results["baseline"]["checkpoint_path"] = ckpt_baseline
    results["baseline"]["dag_depth"] = getattr(baseline_model.config, "dag_depth", 0)
    results["baseline"]["n_params"] = baseline_model.get_num_params()

    results["dag"]["checkpoint_path"] = ckpt_dag
    results["dag"]["dag_depth"] = getattr(dag_model.config, "dag_depth", 0)
    results["dag"]["n_params"] = dag_model.get_num_params()

    # Comprehensive evaluation if data_dir is available
    if data_dir and Path(data_dir).exists():
        data_path = Path(data_dir)
        print(f"\n=== Comprehensive Evaluation using {data_dir} ===")

        print("\nEvaluating baseline model...")
        baseline_results = comprehensive_evaluate(
            baseline_model,
            data_path,
            device,
            eval_iters=eval_iters,
            batch_size=batch_size,
            math_tasks=math_tasks,
            math_max_examples=math_max_examples,
            enable_dag_logging=True,
            generate_text=True,
        )
        results["baseline"].update(baseline_results)

        print("\nEvaluating DAG model...")
        dag_results = comprehensive_evaluate(
            dag_model,
            data_path,
            device,
            eval_iters=eval_iters,
            batch_size=batch_size,
            math_tasks=math_tasks,
            math_max_examples=math_max_examples,
            enable_dag_logging=True,
            generate_text=True,
        )
        results["dag"].update(dag_results)

    # Fallback evaluation using dataset file
    elif dataset_file and Path(dataset_file).exists():
        print(f"\n=== Basic Evaluation using {dataset_file} ===")

        print("Evaluating baseline model...")
        baseline_loss = evaluate_from_dataset_file(baseline_model, dataset_file, device)
        results["baseline"]["dataset_loss"] = baseline_loss

        print("Evaluating DAG model...")
        dag_loss = evaluate_from_dataset_file(dag_model, dataset_file, device)
        results["dag"]["dataset_loss"] = dag_loss

    else:
        print(
            "Warning: No data directory or dataset file provided. Running minimal evaluation."
        )
        # Run math evaluation only
        try:
            print("Running math evaluation on baseline...")
            baseline_math = evaluate_math(
                baseline_model, device, math_tasks, math_max_examples
            )
            results["baseline"].update(
                {f"math_eval_{k}": v for k, v in baseline_math.items()}
            )

            print("Running math evaluation on DAG...")
            dag_math = evaluate_math(dag_model, device, math_tasks, math_max_examples)
            results["dag"].update({f"math_eval_{k}": v for k, v in dag_math.items()})
        except Exception as e:
            print(f"Warning: Math evaluation failed: {e}")

    # Compute comparisons
    print("\n=== Results Comparison ===")
    for key in results["baseline"]:
        if key in results["dag"] and isinstance(results["baseline"][key], (int, float)):
            baseline_val = results["baseline"][key]
            dag_val = results["dag"][key]

            if baseline_val != 0:
                improvement = ((baseline_val - dag_val) / abs(baseline_val)) * 100
                results["comparison"][f"{key}_improvement_pct"] = improvement
            else:
                results["comparison"][f"{key}_difference"] = dag_val - baseline_val

    # Print results
    print(f"\nBaseline Model (dag_depth={results['baseline']['dag_depth']}):")
    for key, value in results["baseline"].items():
        if isinstance(value, (int, float)) and key != "n_params":
            print(f"  {key}: {value:.4f}")
        elif key == "n_params":
            print(f"  {key}: {value:,}")
        elif isinstance(value, str) and len(value) < 100:
            print(f"  {key}: {value}")

    print(f"\nDAG Model (dag_depth={results['dag']['dag_depth']}):")
    for key, value in results["dag"].items():
        if isinstance(value, (int, float)) and key != "n_params":
            print(f"  {key}: {value:.4f}")
        elif key == "n_params":
            print(f"  {key}: {value:,}")
        elif isinstance(value, str) and len(value) < 100:
            print(f"  {key}: {value}")

    print("\nComparison (positive = DAG better):")
    for key, value in results["comparison"].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save results if requested
    if save_results:
        save_path = Path(save_results)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare two model checkpoints with comprehensive evaluation"
    )
    parser.add_argument(
        "--ckpt_baseline", required=True, help="Path to baseline checkpoint"
    )
    parser.add_argument("--ckpt_dag", required=True, help="Path to DAG checkpoint")
    parser.add_argument(
        "--data_dir",
        help="Directory containing train.bin/val.bin for comprehensive evaluation",
    )
    parser.add_argument(
        "--dataset",
        default="tests/math_eval.txt",
        help="Dataset file for basic evaluation (fallback)",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of iterations for loss estimation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--math_tasks",
        nargs="+",
        default=["gsm8k", "svamp"],
        help="Math evaluation tasks",
    )
    parser.add_argument(
        "--math_max_examples", type=int, default=50, help="Max examples per math task"
    )
    parser.add_argument("--save_results", help="Path to save results as JSON")

    args = parser.parse_args()

    results = compare_models_comprehensive(
        ckpt_baseline=args.ckpt_baseline,
        ckpt_dag=args.ckpt_dag,
        data_dir=args.data_dir,
        dataset_file=args.dataset,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        math_tasks=args.math_tasks,
        math_max_examples=args.math_max_examples,
        save_results=args.save_results,
    )

    return results


if __name__ == "__main__":
    main()
