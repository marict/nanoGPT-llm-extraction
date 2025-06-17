import os
from typing import Sequence

import runpod

from python_version_check import check_python_version

check_python_version()

DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 5090"
REPO_URL = "https://github.com/marict/nanoGPT-llm-extraction.git"


def _get_wandb_url(cfg_path: str) -> str:
    """Return the expected Weights & Biases URL for ``cfg_path``."""
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)
    data: dict[str, str] = {}
    try:  # pragma: no cover - best effort for user feedback
        with open(cfg_path, "r") as f:
            exec(f.read(), data)
        project = data.get("wandb_project")
        run_name = data.get("wandb_run_name")
        if project and run_name:
            return f"https://wandb.ai/{project}/{run_name}"
    except Exception:
        pass
    return "https://wandb.ai"


def _resolve_gpu_id(gpu_type: str) -> str:
    """Return the GPU id for ``gpu_type`` which may be a name or id."""
    try:
        gpus = runpod.get_gpus()
    except Exception as exc:  # pragma: no cover - network errors
        raise RunPodError(f"Failed to list GPUs: {exc}") from exc

    for gpu in gpus:
        if gpu_type in {gpu.get("id"), gpu.get("displayName")}:
            return gpu["id"]
    raise RunPodError(f"GPU type '{gpu_type}' not found")


class RunPodError(Exception):
    """Custom exception for RunPod operations."""


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: str | None = None,
) -> str:
    """Launch a RunPod GPU instance and run training automatically."""

    runpod.api_key = (
        api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    )
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set the RUNPOD_API_KEY environment variable"
        )

    args_list = train_args.split()
    wandb_url = "https://wandb.ai"
    if args_list:
        cfg_path = args_list[0]
        wandb_url = _get_wandb_url(cfg_path)
        if not os.path.isabs(cfg_path):
            args_list[0] = f"/workspace/{cfg_path}"
    train_args = " ".join(args_list)

    gpu_type_id = _resolve_gpu_id(gpu_type)

    pod = runpod.create_pod(
        name="daggpt-train",
        image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        start_ssh=False,
        docker_args=(
            f"bash -c '\n"
            f"[ -d repo ] && git -C repo pull || git clone {REPO_URL} repo;\n"
            f"cd repo\n"
            f"pip install -r requirements-dev.txt\n"
            f"python train.py {train_args}\n'"
        )
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")
    print(
        f"Starting training job 'daggpt-train' (pod {pod_id}) on {gpu_type}. "
        f"View logs at {wandb_url}"
    )

    return pod_id


def visualize_dag_attention(
    model,
    tokenizer,
    prompt: str,
    save_path: str = "dag_attention.png",
):
    """Run ``model`` on ``prompt`` and save a DAG attention heatmap."""

    import matplotlib.pyplot as plt
    import torch

    from dag_model import DAGGPT

    if not isinstance(model, DAGGPT):
        raise TypeError("model must be DAGGPT")

    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        _, _, _, dag_info = model(x, return_dag_info=True)

    attn_history: Sequence[torch.Tensor] = dag_info["attn"]
    max_len = max(t.numel() for t in attn_history)
    mat = torch.zeros(len(attn_history), max_len)
    for i, att in enumerate(attn_history):
        mat[i, : att.numel()] = att

    fig, ax = plt.subplots()
    im = ax.imshow(mat.numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Node")
    ax.set_ylabel("DAG Step")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved DAG attention visualization to {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RunPod helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start training pod")
    t.add_argument("config", help="Training config file")
    t.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type name")
    t.add_argument("--api-key", help="RunPod API key")

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(
            args.config,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
        )
