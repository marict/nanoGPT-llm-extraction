import os
import time
from typing import Sequence

import requests
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
    gpu_start = time.time()
    print(f"[{time.time() - gpu_start:.2f}s] Resolving GPU type: {gpu_type}")

    try:
        gpus_start = time.time()
        gpus = runpod.get_gpus()
        print(
            f"[{time.time() - gpu_start:.2f}s] GPU list retrieved in {time.time() - gpus_start:.2f}s"
        )
    except Exception as exc:  # pragma: no cover - network errors
        print(f"[{time.time() - gpu_start:.2f}s] Failed to list GPUs: {exc}")
        raise RunPodError(f"Failed to list GPUs: {exc}") from exc

    search_start = time.time()
    for gpu in gpus:
        if gpu_type in {gpu.get("id"), gpu.get("displayName")}:
            print(
                f"[{time.time() - gpu_start:.2f}s] GPU found in {time.time() - search_start:.2f}s"
            )
            return gpu["id"]

    print(
        f"[{time.time() - gpu_start:.2f}s] GPU search completed in {time.time() - search_start:.2f}s"
    )
    raise RunPodError(f"GPU type '{gpu_type}' not found")


class RunPodError(Exception):
    """Custom exception for RunPod operations."""


def stop_runpod():
    stop_start = time.time()
    print(f"[{time.time() - stop_start:.2f}s] Starting pod stop process")

    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    if pod_id and api_key:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            request_start = time.time()
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            print(
                f"[{time.time() - stop_start:.2f}s] Pod stop request completed in {time.time() - request_start:.2f}s"
            )
            print("Successfully requested pod stop.")
        except Exception as e:
            print(f"[{time.time() - stop_start:.2f}s] Failed to stop pod: {e}")
    else:
        print(
            f"[{time.time() - stop_start:.2f}s] RUNPOD_POD_ID or RUNPOD_API_KEY not set, skipping pod stop."
        )


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: str | None = None,
) -> str:
    """Launch a RunPod GPU instance and run training automatically."""

    service_start = time.time()
    print(f"[{time.time() - service_start:.2f}s] Starting cloud training service")

    # API key setup
    api_start = time.time()
    runpod.api_key = (
        api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    )
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set the RUNPOD_API_KEY environment variable"
        )
    print(
        f"[{time.time() - service_start:.2f}s] API key setup completed in {time.time() - api_start:.2f}s"
    )

    # Argument processing
    args_start = time.time()
    args_list = train_args.split()
    if args_list:
        cfg_path = args_list[0]
        if not os.path.isabs(cfg_path):
            args_list[0] = f"{cfg_path}"
    args_list.append(f"--wandb_project={POD_NAME}")
    train_args = " ".join(args_list)
    print(
        f"[{time.time() - service_start:.2f}s] Argument processing completed in {time.time() - args_start:.2f}s"
    )

    # GPU resolution
    gpu_resolve_start = time.time()
    gpu_type_id = _resolve_gpu_id(gpu_type)
    print(
        f"[{time.time() - service_start:.2f}s] GPU resolution completed in {time.time() - gpu_resolve_start:.2f}s"
    )

    # Docker args preparation
    docker_start = time.time()
    docker_args = (
        f"bash -c 'set -e ; "
        f"cd /workspace ; "
        f"[ -d repo ] && git -C repo pull || git clone {REPO_URL} repo ; "
        f"cd repo ; "
        f"apt-get update && apt-get install -y tree ; "
        f"echo === Directory Structure === && tree ; "
        f"echo === Current Directory === && pwd ; "
        f"pip install -q -r requirements-dev.txt ; "
        f"python train.py {train_args} 2>&1 | "
        f"tee /workspace/train_$(date +%Y%m%d_%H%M%S).log ; "
        f"tail -f /dev/null'"
    )
    print(
        f"[{time.time() - service_start:.2f}s] Docker args preparation completed in {time.time() - docker_start:.2f}s"
    )

    # Pod creation
    pod_start = time.time()
    print(
        f"[{time.time() - service_start:.2f}s] Creating pod with GPU type: {gpu_type}"
    )
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        min_vcpu_count=8,
        min_memory_in_gb=128,
        volume_in_gb=180,  # persists across stops
        container_disk_in_gb=160,  # wiped on stop
        env={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
        },
        start_ssh=False,
        docker_args=docker_args,
    )
    print(
        f"[{time.time() - service_start:.2f}s] Pod creation API call completed in {time.time() - pod_start:.2f}s"
    )

    # Pod ID extraction
    id_start = time.time()
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")
    print(
        f"[{time.time() - service_start:.2f}s] Pod ID extraction completed in {time.time() - id_start:.2f}s"
    )

    print(
        f"[{time.time() - service_start:.2f}s] Starting training job '{POD_NAME}' (pod {pod_id}) on {gpu_type}."
    )
    print(
        f"[{time.time() - service_start:.2f}s] Total service startup time: {time.time() - service_start:.2f}s"
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

    main_start = time.time()
    print(f"[{time.time() - main_start:.2f}s] Starting RunPod service main function")

    parser_start = time.time()
    parser = argparse.ArgumentParser(description="RunPod helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start training pod")
    t.add_argument("config", help="Training config file")
    t.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type name")
    t.add_argument("--api-key", help="RunPod API key")

    args = parser.parse_args()
    print(
        f"[{time.time() - main_start:.2f}s] Argument parsing completed in {time.time() - parser_start:.2f}s"
    )

    if args.cmd == "train":
        start_cloud_training(
            args.config,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
        )
