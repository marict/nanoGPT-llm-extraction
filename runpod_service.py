import os
from typing import Sequence

import runpod

from python_version_check import check_python_version

check_python_version()

# Available GPU types for RunPod:
# - NVIDIA A40
# - NVIDIA A100 80GB PCIe
# - NVIDIA A100-SXM4-80GB
# - NVIDIA B200
# - NVIDIA H100 PCIe
# - NVIDIA H100 80GB HBM3
# - NVIDIA H100 NVL
# - NVIDIA H200
# - NVIDIA L40
# - NVIDIA L40S
# - NVIDIA RTX 6000 Ada Generation
# - NVIDIA RTX A6000
# - NVIDIA RTX PRO 6000 Blackwell Workstation Edition

# DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 5090"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
REPO_URL = "https://github.com/marict/nanoGPT-llm-extraction.git"
POD_NAME = "daggpt-train"


def _get_wandb_url(cfg_path: str) -> str:
    """Return the expected Weights & Biases URL for ``cfg_path``."""
    data: dict[str, str] = {}
    try:  # pragma: no cover - best effort for user feedback
        with open(cfg_path, "r") as f:
            exec(f.read(), data)
        project = data.get("wandb_project")
        run_name = POD_NAME
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
            args_list[0] = f"{cfg_path}"
    args_list.append(f"--wandb_project={POD_NAME}")
    train_args = " ".join(args_list)

    gpu_type_id = _resolve_gpu_id(gpu_type)
    docker_args = (
        f"bash -c '[ -d repo ] && git -C repo pull || git clone {REPO_URL} repo && "
        f"cd repo && "
        f"apt-get update && apt-get install -y tree && "
        f"echo === Directory Structure === && tree && "
        f"echo === Current Directory === && pwd && "
        f"echo === Config File Location === && ls -la config/train_default.py && "
        f"pip install -q -r requirements-dev.txt && "
        f"python train.py {train_args}'"
    )
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        gpu_count=1,  # one A100-40GB or A6000
        min_vcpu_count=8,  # ≥ 8 cores
        min_memory_in_gb=128,  # 128 GB RAM
        volume_in_gb=80,  # persistent storage
        container_disk_in_gb=40,  # docker-layer scratch (adjust if you like)
        start_ssh=False,
        docker_args=docker_args,
        env={"WANDB_API_KEY": os.getenv("WANDB_API_KEY", "")},
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")
    print(f"Starting training job '{POD_NAME}' (pod {pod_id}) on {gpu_type}. ")

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
