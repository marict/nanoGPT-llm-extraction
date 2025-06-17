import os
import subprocess
from typing import Sequence
try:
    import runpod
    from runpod.cli.utils.rp_info import get_pod_ssh_ip_port
    from runpod.cli.utils.ssh_cmd import SSHConnection
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    import types

    runpod = types.SimpleNamespace(create_pod=None, api_key=None)
    get_pod_ssh_ip_port = None
    SSHConnection = None
from python_version_check import check_python_version

check_python_version()

DEFAULT_GPU_TYPE = "NVIDIA A100-SXM4-40GB"

class RunPodError(Exception):
    """Custom exception for RunPod operations."""


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: str | None = None,
) -> str:
    """Launch a RunPod GPU instance and run training via SSH."""
    if runpod is None or get_pod_ssh_ip_port is None or SSHConnection is None:
        raise RunPodError(
            "runpod package is required to start cloud training"
        )

    runpod.api_key = api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set the RUNPOD_API_KEY environment variable"
        )

    args_list = train_args.split()
    if args_list:
        cfg_path = args_list[0]
        if not os.path.isabs(cfg_path):
            args_list[0] = f"/workspace/{cfg_path}"
    train_args = " ".join(args_list)

    pod = runpod.create_pod(
        name="daggpt-train",
        image_name="runpod/stack",
        gpu_type_id=gpu_type,
        ports="22/tcp",
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")
    print(f"Created pod {pod_id}")

    # wait for pod to start and get ssh details
    ip_addr, port = get_pod_ssh_ip_port(pod_id)

    ssh = SSHConnection(pod_id)
    cmd = f"cd /workspace && python train.py {train_args}"
    ssh.run_commands([cmd])
    ssh.close()
    return pod_id


def visualize_dag_attention(
    model,
    tokenizer,
    prompt: str,
    save_path: str = "dag_attention.png",
):
    """Run ``model`` on ``prompt`` and save a DAG attention heatmap."""

    import torch
    import matplotlib.pyplot as plt
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
