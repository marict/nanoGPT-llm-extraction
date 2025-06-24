"""
Comprehensive evaluation utilities for GPT models.

This module contains all evaluation functions extracted from the training loop
to allow reuse in scripts like compare_checkpoints.py.
"""

import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
import torch

import run_math_eval
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig


def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    ctx: nullcontext | torch.autocast,
) -> Dict[str, torch.Tensor]:
    """Return mean loss over <eval_iters> batches for train/val splits."""
    out: Dict[str, torch.Tensor] = {}
    model.eval()
    print("Estimating loss...")
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch_fn(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def evaluate_math(
    model: torch.nn.Module, device: str, tasks: List[str] = None, max_examples: int = 50
) -> Dict[str, float]:
    """Run math evaluation during training."""
    if tasks is None:
        tasks = ["gsm8k", "svamp"]  # Default to both GSM8K and SVAMP during training

    try:
        print(
            f"Running math evaluation on {tasks} (max {max_examples} examples each)..."
        )
        scores = run_math_eval.run_eval(
            model, device, tasks=tasks, max_examples=max_examples
        )
        return scores
    except Exception as e:
        print(f"Warning: Math evaluation failed: {e}")
        return {task: -1.0 for task in tasks}


def setup_tokenizer(data_dir: Path) -> Tuple[Optional[Callable], Optional[Callable]]:
    """Setup encoder/decoder functions for text generation."""
    meta_path = data_dir / "meta.pkl"
    encode = decode = None

    # Try to load from meta.pkl first
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if "stoi" in meta and "itos" in meta:
                stoi, itos = meta["stoi"], meta["itos"]
                encode = lambda s: [stoi[c] for c in s]
                decode = lambda l: "".join([itos[i] for i in l])
                return encode, decode
        except Exception as e:
            print(f"Warning: Failed to load meta.pkl: {e}")

    # Fallback to GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def generate_sample_text(
    model: torch.nn.Module,
    encode: Callable,
    decode: Callable,
    device: str,
    prompt: str = "Two plus 5 is equal to: ",
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate a sample text from the model for quality assessment."""
    try:
        encoded = encode(prompt)
        prompt_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            return decode(generated[0].cpu().tolist())
    except Exception as e:
        return f"Error: {str(e)}"


def create_batch_function(
    data_dir: Path, batch_size: int, block_size: int, device: str
) -> Tuple[Callable[[str], Tuple[torch.Tensor, torch.Tensor]], np.dtype]:
    """Create a batch function for the given dataset."""
    # Load meta to determine dtype
    meta_path = data_dir / "meta.pkl"
    meta_dtype = np.uint16
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_dtype = np.uint8 if meta.get("byte_level", False) else np.uint16
        except Exception:
            pass

    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one batch of tokens for <split>."""
        file = "train.bin" if split == "train" else "val.bin"
        data = np.memmap(data_dir / file, dtype=meta_dtype, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
                for i in ix
            ]
        )
        if device == "cuda":
            x, y = (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, meta_dtype


def comprehensive_evaluate(
    model: torch.nn.Module,
    data_dir: Path,
    device: str,
    eval_iters: int = 200,
    batch_size: int = 8,
    block_size: int = 1024,
    math_tasks: List[str] = None,
    math_max_examples: int = 50,
    enable_dag_logging: bool = True,
    generate_text: bool = True,
) -> Dict[str, float]:
    """
    Run comprehensive evaluation including all metrics from training loop.

    Args:
        model: The model to evaluate
        data_dir: Directory containing train.bin and val.bin
        device: Device to run evaluation on
        eval_iters: Number of iterations for loss estimation
        batch_size: Batch size for evaluation
        block_size: Block size for evaluation
        math_tasks: Math evaluation tasks (default: ["gsm8k", "svamp"])
        math_max_examples: Max examples per math task
        enable_dag_logging: Whether to compute DAG-specific metrics
        generate_text: Whether to generate sample text

    Returns:
        Dictionary containing all evaluation metrics
    """
    if math_tasks is None:
        math_tasks = ["gsm8k", "svamp"]

    # Setup evaluation context
    ptdtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    results = {}

    # 1. Loss estimation
    if (data_dir / "train.bin").exists() and (data_dir / "val.bin").exists():
        try:
            get_batch, _ = create_batch_function(
                data_dir, batch_size, block_size, device
            )
            losses = estimate_loss(model, eval_iters, get_batch, ctx)
            results.update(
                {"train_loss": losses["train"].item(), "val_loss": losses["val"].item()}
            )
        except Exception as e:
            print(f"Warning: Loss estimation failed: {e}")
            results.update({"train_loss": -1.0, "val_loss": -1.0})

    # 2. Math evaluation
    try:
        math_scores = evaluate_math(model, device, math_tasks, math_max_examples)
        for task, score in math_scores.items():
            results[f"math_eval_{task}"] = score
    except Exception as e:
        print(f"Warning: Math evaluation failed: {e}")
        for task in math_tasks:
            results[f"math_eval_{task}"] = -1.0

    # 3. DAG-specific metrics (if applicable)
    if (
        enable_dag_logging
        and hasattr(model, "config")
        and getattr(model.config, "dag_depth", 0) > 0
    ):
        try:
            dag_logger = DAGLogger()
            dag_logger.setup_gradient_tracking(model)
            extra_vals = dag_logger.get_extra_vals(model)
            results.update(extra_vals)
        except Exception as e:
            print(f"Warning: DAG logging failed: {e}")

    # 4. Text generation sample
    if generate_text:
        try:
            encode, decode = setup_tokenizer(data_dir)
            sample_text = generate_sample_text(model, encode, decode, device)
            results["generated_sample"] = sample_text
            print(f"Generated sample: {sample_text}")
        except Exception as e:
            print(f"Warning: Text generation failed: {e}")
            results["generated_sample"] = f"Error: {str(e)}"

    return results


def load_checkpoint(ckpt_path: str, device: str):
    """Load a model checkpoint and return the configured model."""

    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt["model_args"]

    # Ensure dag_depth is set (default to 0 for standard GPT)
    model_args.setdefault("dag_depth", 0)

    config = GPTConfig(**model_args)
    model = GPT(config)

    # Remove unwanted prefix from state dict keys
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ckpt


def evaluate_from_dataset_file(
    model: torch.nn.Module, dataset_file: str, device: str, encode_fn: Callable = None
) -> float:
    """Evaluate model on a simple dataset file (like the original compare_checkpoints)."""
    if encode_fn is None:
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = enc.encode

    losses = []

    # Get model vocab size to filter out-of-range tokens
    vocab_size = (
        getattr(model.config, "vocab_size", 50304)
        if hasattr(model, "config")
        else 50304
    )

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            # Parse line (assuming Q: ... A: ... format)
            if "\t" in line:
                q, a = line.split("\t", 1)
                q = q.replace("Q:", "").strip()
                a = a.replace("A:", "").strip()
            else:
                # Simple text line
                q, a = "", line.strip()

            text = q + " " + a if q else a
            ids = encode_fn(text)

            # Filter out tokens that are out of range for this model
            ids = [id for id in ids if id < vocab_size]

            if len(ids) < 2:
                continue

            x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]
            y = torch.tensor(ids[1:], dtype=torch.long, device=device)[None, :]

            with torch.no_grad():
                _, loss = model(x, targets=y)
            losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("inf")
