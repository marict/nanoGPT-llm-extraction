import os
import time
import runpod


DEFAULT_IMAGE = "runpod/pytorch:2.2.1-cuda12.1-devel"
DEFAULT_GPU = "NVIDIA A100 40GB PCIe"


class RunpodError(Exception):
    """Custom exception for RunPod operations."""


def start_cloud_training(config_path: str, gpu_type: str = DEFAULT_GPU) -> str:
    """Launch a training job on RunPod and stream status to the console.

    Args:
        config_path: Path to the training config file.
        gpu_type: GPU type to request.

    Returns:
        The created pod ID.
    """
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise RunpodError("RUNPOD_API_KEY environment variable is required")

    runpod.api_key = api_key

    pod = runpod.create_pod(
        name="nanogpt-training",
        image_name=DEFAULT_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        start_ssh=True,
        docker_args=f"python train.py {config_path}",
    )
    pod_id = pod.get("id")
    if not pod_id:
        raise RunpodError("Failed to create pod")

    print(f"Created pod {pod_id}")

    while True:
        info = runpod.get_pod(pod_id)
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

    runpod.api_key = api_key
    endpoint = runpod.Endpoint(endpoint_id)
    result = endpoint.run_sync({"input": prompt})
    output = result.get("output", result)
    print(output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RunPod helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start training job")
    t.add_argument("config", help="Training config file")
    t.add_argument("--gpu", default=DEFAULT_GPU, help="GPU type id")

    i = sub.add_parser("infer", help="Run inference")
    i.add_argument("prompt", help="Prompt text")
    i.add_argument("--endpoint", help="RunPod endpoint id")

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(args.config, args.gpu)
    else:
        run_inference(args.prompt, args.endpoint)
