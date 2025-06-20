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
# DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_GPU_TYPE = (
    "NVIDIA RTX 6000 Ada Generation"  # What is available in the WA network volume
)
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


def stop_runpod(pod_id: str | None = None, api_key: str | None = None) -> bool:
    """
    Stop the active RunPod instance.

    Returns
    -------
    bool
        True  – stop signal accepted
        False – stop failed / not attempted
    """
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")

    if not pod_id or not api_key:
        print("RUNPOD_POD_ID or RUNPOD_API_KEY not set – skipping pod stop.")
        return False

    # ------------------------------------------------------------------ #
    # 1 Try the Python SDK (cleanest, no hard-coding of REST paths)
    # ------------------------------------------------------------------ #
    try:
        runpod.api_key = api_key
        # stop_pod is available in runpod>=0.9.0
        if hasattr(runpod, "stop_pod"):
            runpod.stop_pod(pod_id)  # raises on error
            print("Successfully requested pod stop (SDK).")
            return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"[stop_runpod] SDK method failed: {exc}.  "
            "Falling back to raw REST call …"
        )

    # ------------------------------------------------------------------ #
    # 2 Fallback – direct REST POST
    # ------------------------------------------------------------------ #
    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print("Successfully requested pod stop (REST).")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to stop pod: {exc}")
        return False


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
    # Docker args preparation
    docker_start = time.time()

    # Create an inline script that clones the repo first, then runs the setup script
    inline_script = (
        f"apt-get update && apt-get install -y git && "
        f"cd /workspace && "
        f"( [ -d repo/.git ] && git -C repo pull || git clone https://github.com/marict/nanoGPT-llm-extraction.git repo ) && "
        f"bash /workspace/repo/scripts/container_setup.sh {train_args}"
    )
    docker_args = f"bash -c '{inline_script}'"
    print(
        f"[{time.time() - docker_start:.2f}s] Docker args preparation completed in {time.time() - docker_start:.2f}s"
    )
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        min_vcpu_count=8,
        min_memory_in_gb=64,
        volume_in_gb=1000,  # persists across stops
        container_disk_in_gb=1000,  # wiped on stop
        network_volume_id="xfv5wps96a",
        env={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
        },
        start_ssh=False,
        docker_args=docker_args,
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
        _, _, dag_info = model(x, return_dag_info=True)

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
