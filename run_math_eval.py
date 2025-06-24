"""Run arithmetic benchmarks (GSM8K, SVAMP) directly on live model."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, Sequence

import datasets
import tiktoken
import torch
from tqdm import tqdm

NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")
DEFAULT_TASKS: Sequence[str] = ("gsm8k", "svamp")


def _extract_number(text: str) -> str:
    """Extract first number string from model output."""
    m = NUMBER_RE.search(text)
    return m.group(0) if m else ""


def _make_prompt(task: str, ex: dict) -> str:
    if task == "gsm8k":
        return f"{ex['question'].strip()}\nAnswer:"
    if task == "svamp":
        # ChilleD/SVAMP dataset format uses 'Body' and 'Question' fields
        return f"{ex['Body'].strip()} {ex['Question'].strip()}\nAnswer:"
    raise ValueError(f"Unsupported task {task}")


def _gold_answer(task: str, ex: dict) -> str:
    if task == "gsm8k":
        return ex["answer"].split("####")[-1].strip()
    if task == "svamp":
        # ChilleD/SVAMP dataset format uses 'Answer' field
        return str(ex["Answer"]).strip()
    raise ValueError(f"Unsupported task {task}")


def _load_dataset(task: str, split: str = "test"):
    """Load dataset with proper configuration."""
    if task == "gsm8k":
        return datasets.load_dataset("gsm8k", "main", split=split)
    elif task == "svamp":
        # Use the correct SVAMP dataset from Hugging Face
        return datasets.load_dataset("ChilleD/SVAMP", split=split)
    else:
        raise ValueError(f"Unsupported task {task}")


def _attach_tokenizer_methods(
    model: torch.nn.Module, device: str | torch.device
) -> None:
    """Attach encode/decode methods to model using appropriate tokenizer."""
    # Try to find dataset-specific tokenizer first
    meta_path = None
    if hasattr(model, "config") and hasattr(model.config, "dataset"):
        meta_path = Path("data") / model.config.dataset / "meta.pkl"

    # Look for any meta.pkl file if not found
    if not meta_path or not meta_path.exists():
        for data_path in Path("data").glob("*/meta.pkl"):
            meta_path = data_path
            break

    # Setup tokenizer
    if meta_path and meta_path.exists():
        # Use dataset-specific tokenizer
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("tokenizer") == "gpt2" or "stoi" not in meta:
            # GPT-2 tokenizer
            enc = tiktoken.get_encoding("gpt2")
            model.encode = lambda s: torch.tensor(
                enc.encode(s, allowed_special={"<|endoftext|>"}),
                dtype=torch.long,
                device=device,
            )
            model.decode = lambda l: enc.decode(
                l.tolist() if isinstance(l, torch.Tensor) else l
            )
        else:
            # Character-level tokenizer
            stoi, itos = meta["stoi"], meta["itos"]
            model.encode = lambda s: torch.tensor(
                [stoi[c] for c in s], dtype=torch.long, device=device
            )
            model.decode = lambda l: "".join(
                [itos[i] for i in (l.tolist() if isinstance(l, torch.Tensor) else l)]
            )
    else:
        # Default to GPT-2 tokenizer
        enc = tiktoken.get_encoding("gpt2")
        model.encode = lambda s: torch.tensor(
            enc.encode(s, allowed_special={"<|endoftext|>"}),
            dtype=torch.long,
            device=device,
        )
        model.decode = lambda l: enc.decode(
            l.tolist() if isinstance(l, torch.Tensor) else l
        )


def run_eval(
    model: torch.nn.Module,
    device: str | torch.device,
    tasks: Sequence[str] | None = None,
    max_examples: int | None = None,
    max_new_tokens: int = 32,
) -> Dict[str, float]:
    """Evaluate model on GSM8K/SVAMP by exact match numeric accuracy."""
    model.eval()
    model.to(device)

    # Attach encode/decode methods if they don't exist
    if not hasattr(model, "encode") or not hasattr(model, "decode"):
        _attach_tokenizer_methods(model, device)

    tasks = list(tasks or DEFAULT_TASKS)
    scores: Dict[str, float] = {}

    for task in tasks:
        try:
            ds = _load_dataset(task)
            if max_examples:
                ds = ds.select(range(min(max_examples, len(ds))))

            correct = 0
            failed_examples = False
            for ex in tqdm(ds, desc=f"eval:{task}", leave=False):
                try:
                    prompt = _make_prompt(task, ex)
                    input_ids = model.encode(prompt).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out_ids = model.generate(
                            input_ids, max_new_tokens=max_new_tokens, temperature=1.0
                        )[0]

                    prediction = _extract_number(
                        model.decode(out_ids[len(input_ids[0]) :])
                    )
                    gold = _extract_number(_gold_answer(task, ex))
                    correct += int(prediction == gold)
                except Exception as e:
                    print(f"Warning: Error processing example in {task}: {e}")
                    failed_examples = True
                    # Continue with next example
                    continue

            # If any examples failed, return -1.0 to signal evaluation issues
            if failed_examples:
                scores[task] = -1.0
            else:
                scores[task] = correct / len(ds) if len(ds) > 0 else 0.0
        except Exception as e:
            print(f"Error evaluating task {task}: {e}")
            scores[task] = -1.0

    model.train()
    return scores


def main():
    """CLI interface for running math evaluation."""
    parser = argparse.ArgumentParser(
        description="Run math evaluation on a trained model"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Tasks to evaluate (default: gsm8k svamp)",
    )
    parser.add_argument("--max-examples", type=int, help="Maximum examples per task")
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Determine model type based on checkpoint
    model_args = checkpoint["model_args"]
    from dag_model import GPT, GPTConfig

    # Ensure dag_depth is set (default to 0 for standard GPT)
    model_args.setdefault("dag_depth", 0)
    config = GPTConfig(**model_args)
    model = GPT(config)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    # Run evaluation
    print(f"Running evaluation on tasks: {args.tasks}")
    scores = run_eval(
        model,
        args.device,
        tasks=args.tasks,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
    )

    # Print results
    print("\nResults:")
    for task, score in scores.items():
        print(f"{task}: {score:.4f}")

    avg_score = sum(scores.values()) / len(scores)
    print(f"Average: {avg_score:.4f}")


if __name__ == "__main__":
    main()
