import os
from typing import Sequence

import runpod

from python_version_check import check_python_version

check_python_version()

DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 5090"


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
    overrides = []
    if args_list:
        cfg_path = args_list[0]
        overrides = [arg for arg in args_list[1:] if arg.startswith("--")]
        if not os.path.isabs(cfg_path):
            args_list[0] = f"/workspace/{cfg_path}"
    train_args = " ".join(args_list)

    wandb_project = wandb_run_name = None
    try:  # pragma: no cover - defensive
        from train import (
            TrainConfig,
            load_config_file,
            update_config,
            apply_overrides,
        )

        cfg = TrainConfig()
        update_config(cfg, load_config_file(cfg_path))
        apply_overrides(cfg, overrides)
        wandb_project = cfg.wandb_project
        wandb_run_name = cfg.wandb_run_name
    except Exception:
        pass

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise RunPodError(
            "WANDB_API_KEY environment variable must be set for wandb logging"
        )

    import wandb

    run_id = wandb.util.generate_id()
    os.environ["WANDB_RUN_ID"] = run_id
    try:
        api = wandb.Api()
        entity = api.viewer().get("entity")
    except Exception:  # pragma: no cover - network/auth issues
        entity = None

    wandb_url = f"https://wandb.ai/{(entity + '/' if entity else '')}{wandb_project}/runs/{run_id}"

    gpu_type_id = _resolve_gpu_id(gpu_type)

    pod = runpod.create_pod(
        name="daggpt-train",
        image_name="runpod/stack",
        gpu_type_id=gpu_type_id,
        start_ssh=False,
        docker_args=f"python train.py {train_args}",
        env={"WANDB_API_KEY": wandb_api_key, "WANDB_RUN_ID": run_id},
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")
    print(
        f"Starting training job in pod {pod_id} on {gpu_type}. "
        f"Run name: {wandb_run_name}. \nW&B URL: {wandb_url}"
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
