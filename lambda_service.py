import os
import subprocess
import time
from typing import Sequence

import requests

from python_version_check import check_python_version

check_python_version()

API_BASE = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_INSTANCE_TYPE = "gpu_1x_a10"
DEFAULT_REGION = "us-east-1"


class LambdaError(Exception):
    """Custom exception for Lambda Labs operations."""


def _get_headers(api_key: str | None = None) -> dict[str, str]:
    api_key = api_key or os.getenv("LAMBDA_API_KEY")
    if not api_key:
        raise LambdaError(
            "Lambda API key is required. Provide via --api-key or set the LAMBDA_API_KEY environment variable"
        )
    return {"Authorization": f"Bearer {api_key}"}


def start_cloud_training(
    train_args: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    region: str = DEFAULT_REGION,
    ssh_key: str | None = None,
    api_key: str | None = None,
) -> str:
    """Launch a training instance on Lambda Labs and stream status."""
    ssh_key = ssh_key or os.getenv("LAMBDA_SSH_KEY")
    if not ssh_key:
        raise LambdaError("SSH key name must be provided or LAMBDA_SSH_KEY set")

    headers = _get_headers(api_key)

    args_list = train_args.split()
    if args_list:
        cfg_path = args_list[0]
        if not os.path.isabs(cfg_path):
            args_list[0] = f"/workspace/{cfg_path}"
    train_args = " ".join(args_list)

    user_data = (
        "#!/bin/bash\n"
        "curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash\n"
    )

    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key],
        "user_data": user_data,
    }

    resp = requests.post(
        f"{API_BASE}/instance-operations/launch", json=payload, headers=headers
    )
    if resp.status_code // 100 != 2:
        raise LambdaError(f"Failed to launch instance: {resp.text}")

    data = resp.json().get("data", {})
    instance_ids = data.get("instance_ids")
    if not instance_ids:
        raise LambdaError("Lambda API did not return an instance id")
    instance_id = instance_ids[0]
    print(f"Launched instance {instance_id}")

    ip_addr = None
    while True:
        stat = requests.get(f"{API_BASE}/instances/{instance_id}", headers=headers)
        if stat.status_code // 100 != 2:
            raise LambdaError("Lambda API returned no status information")
        info = stat.json().get("data", {})
        status = info.get("status")
        ip_addr = (
            ip_addr
            or info.get("ip")
            or info.get("ip_address")
            or info.get("ipv4")
            or info.get("ip_addresses", {}).get("public")
        )
        print(f"Instance status: {status}")
        if status == "running" and ip_addr:
            break
        if status in {"terminated", "terminating", "unhealthy"}:
            raise LambdaError("Instance failed to start")
        time.sleep(10)

    if not ip_addr:
        raise LambdaError("Could not determine instance IP address")

    ssh_cmd = [
        "ssh",
        f"ubuntu@{ip_addr}",
        f"cd /workspace && python train.py {train_args}",
    ]
    subprocess.run(ssh_cmd, check=True)

    return instance_id


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

    parser = argparse.ArgumentParser(description="Lambda Labs helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start training instance")
    t.add_argument("config", help="Training config file")
    t.add_argument(
        "--instance", default=DEFAULT_INSTANCE_TYPE, help="Instance type name"
    )
    t.add_argument("--region", default=DEFAULT_REGION, help="Region name")
    t.add_argument("--ssh-key", help="SSH key name")
    t.add_argument("--api-key", help="Lambda API key")

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(
            args.config,
            instance_type=args.instance,
            region=args.region,
            ssh_key=args.ssh_key,
            api_key=args.api_key,
        )
