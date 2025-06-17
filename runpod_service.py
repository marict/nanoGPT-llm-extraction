import os
import time
from typing import Sequence

from python_version_check import check_python_version

check_python_version()

runpod = None


def _get_runpod():
    """Lazy import for the ``runpod`` module."""
    global runpod
    if runpod is None:
        import runpod as rp_mod  # type: ignore
        runpod = rp_mod
    return runpod


DEFAULT_IMAGE = "runpod/pytorch:2.2.1-cuda12.1-devel"
DEFAULT_GPU = "NVIDIA A100 40GB PCIe"


class RunpodError(Exception):
    """Custom exception for RunPod operations."""


def start_cloud_training(
    train_args: str, gpu_type: str = DEFAULT_GPU, api_key: str | None = None
) -> str:
    """Launch a training job on RunPod and stream status to the console.

    Args:
        train_args: Arguments to pass to ``train.py`` in the pod.
        gpu_type: GPU type to request.

    Returns:
        The created pod ID.
    """
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise RunpodError("RunPod API key is required")

    rp = _get_runpod()
    rp.api_key = api_key

    pod = rp.create_pod(
        name="nanogpt-training",
        image_name=DEFAULT_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        start_ssh=True,
        docker_args=f"python train.py {train_args}",
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunpodError("Failed to create pod")

    print(f"Created pod {pod_id}")

    while True:
        info = rp.get_pod(pod_id)
        status = info.get("desiredStatus") or info.get("podStatus") or info.get("state")
        print(f"Training status: {status}")
        if status in {"COMPLETED", "STOPPED", "FAILED", "TERMINATED"}:
            break
        time.sleep(10)

    return pod_id


def run_inference(prompt: str, endpoint_id: str | None = None):
    """Run inference on a RunPod endpoint and print the output."""
    endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
    if not endpoint_id:
        raise RunpodError("Endpoint id must be provided or RUNPOD_ENDPOINT_ID set")

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise RunpodError("RUNPOD_API_KEY environment variable is required")

    rp = _get_runpod()
    rp.api_key = api_key
    endpoint = rp.Endpoint(endpoint_id)
    result = endpoint.run_sync({"input": prompt})
    output = result.get("output", result)
    print(output)
    return output


def visualize_dag_attention(
    model,
    tokenizer,
    prompt: str,
    save_path: str = "dag_attention.png",
):
    """Run ``model`` on ``prompt`` and save a DAG attention heatmap.

    Args:
        model: ``DAGGPT`` instance to run.
        tokenizer: Tokenizer used to encode the prompt.
        prompt: Input text to the model.
        save_path: Where to save the generated plot.

    Returns:
        The path to the saved image.
    """

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

    t = sub.add_parser("train", help="Start training job")
    t.add_argument("config", help="Training config file")
    t.add_argument("--gpu", default=DEFAULT_GPU, help="GPU type id")
    t.add_argument("--api-key", help="RunPod API key")

    i = sub.add_parser("infer", help="Run inference")
    i.add_argument("prompt", help="Prompt text")
    i.add_argument("--endpoint", help="RunPod endpoint id")

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(args.config, args.gpu, api_key=args.api_key)
    else:
        run_inference(args.prompt, args.endpoint)
