import argparse
import os
import re
import subprocess
import time

import requests
import runpod

import wandb
from python_version_check import check_python_version

check_python_version()

# Available GPU types for RunPod:
# - NVIDIA A40, A100 80GB PCIe, A100-SXM4-80GB, B200
# - NVIDIA H100 PCIe, H100 80GB HBM3, H100 NVL, H200
# - NVIDIA L40, L40S, RTX 6000 Ada Generation, RTX A6000
# - NVIDIA RTX PRO 6000 Blackwell Workstation Edition

DEFAULT_GPU_TYPE = "NVIDIA RTX 6000 Ada Generation"  # Available in WA network volume
REPO_URL = "https://github.com/marict/nanoGPT-llm-extraction.git"


class RunPodError(Exception):
    """Custom exception for RunPod operations."""


def _validate_note(note: str) -> None:
    """Validate note contains only safe characters."""
    if not note:
        return

    invalid_chars = re.findall(r"[^A-Za-z0-9_\s-]", note)
    if invalid_chars:
        unique_invalid = sorted(set(invalid_chars))
        raise ValueError(
            f"Note contains invalid characters: {unique_invalid}. "
            "Only letters, numbers, spaces, hyphens, and underscores are allowed."
        )


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


def _extract_config_name(config_path: str) -> str:
    """Extract the 'name' field from a training config file."""
    try:
        data: dict[str, str] = {}
        with open(config_path, "r") as f:
            exec(f.read(), data)
        return data.get("name", "daggpt-train")
    except Exception:
        return "daggpt-train"


def _build_training_command(
    train_args: str, keep_alive: bool, note: str | None, wandb_run_id: str | None
) -> str:
    """Build the complete training command with all flags."""
    cmd = train_args

    if keep_alive:
        cmd += " --keep-alive"

    if note:
        escaped_note = note.replace('"', '\\"')
        cmd += f' --note="{escaped_note}"'

    if wandb_run_id:
        cmd += f" --wandb-run-id={wandb_run_id}"

    return cmd


def _create_docker_script(training_command: str) -> str:
    """Create the Docker startup script for the RunPod instance."""
    return (
        f"apt-get update && apt-get install -y git && "
        f"cd /workspace && "
        f"( [ -d repo/.git ] && git -C repo pull || git clone {REPO_URL} repo ) && "
        f"bash /workspace/repo/scripts/container_setup.sh {training_command}"
    )


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: str | None = None,
    keep_alive: bool = False,
    note: str | None = None,
) -> str:
    """Launch a RunPod GPU instance and run training automatically."""

    # Validate inputs
    _validate_note(note)

    # Set up RunPod API
    runpod.api_key = (
        api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    )
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set RUNPOD_API_KEY"
        )

    # Parse config and extract names
    args_list = train_args.split()
    if args_list and not os.path.isabs(args_list[0]):
        args_list[0] = f"{args_list[0]}"

    config_path = args_list[0] if args_list else None
    pod_name = _extract_config_name(config_path) if config_path else "daggpt-train"
    project_name = pod_name
    train_args = " ".join(args_list)

    # Create RunPod instance
    gpu_type_id = _resolve_gpu_id(gpu_type)

    # Build initial training command (without wandb run ID)
    initial_command = _build_training_command(train_args, keep_alive, note, None)
    docker_script = _create_docker_script(initial_command)

    try:
        pod = runpod.create_pod(
            name=pod_name,
            image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            gpu_type_id=gpu_type_id,
            gpu_count=1,
            min_vcpu_count=8,
            min_memory_in_gb=64,
            volume_in_gb=1000,
            container_disk_in_gb=1000,
            network_volume_id="tvi2olc54y",
            env={
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "HF_HOME": "/workspace/.cache/huggingface",
                "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
                "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
            },
            start_ssh=False,
            docker_args=f"bash -c '{docker_script}'",
        )
    except runpod.error.QueryError as exc:  # type: ignore[attr-defined]
        print("RunPod API QueryError:", exc)
        raise

    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")

    # Initialize W&B with correct run name
    wandb_run_name = pod_id if not note else f"{pod_id} - {note}"
    wandb_result = init_local_wandb_and_open_browser(project_name, wandb_run_name)

    if wandb_result:
        wandb_url, wandb_run_id = wandb_result
        print(f"W&B run created: {wandb_url}")
        print(f"Remote training will resume W&B run: {wandb_run_id}")

    print(f"Starting training job '{pod_name}' (pod {pod_id}) on {gpu_type}")
    return pod_id


def init_local_wandb_and_open_browser(
    project_name: str, run_name: str
) -> tuple[str, str] | None:
    """Initialize W&B project locally and open the run URL in browser."""
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set, skipping local W&B initialization")
        return None

    try:
        # Initialize W&B run
        run = wandb.init(
            project=project_name,
            name=run_name,
            tags=["runpod", "remote-training"],
            notes=f"Remote training on RunPod instance {run_name}",
        )

        wandb_url = run.url + "/logs"
        wandb_run_id = run.id

        # Try to open in browser
        _open_browser(wandb_url)

        return (wandb_url, wandb_run_id)

    except Exception as e:
        print(f"Failed to initialize W&B locally: {e}")
        return None


def _open_browser(url: str) -> None:
    """Attempt to open URL in Chrome browser."""
    chrome_commands = [
        "google-chrome",  # Linux
        "google-chrome-stable",  # Linux alternative
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "chrome",  # Windows/generic
        "chromium",  # Linux alternative
        "chromium-browser",  # Linux alternative
    ]

    for chrome_cmd in chrome_commands:
        try:
            subprocess.run(
                [chrome_cmd, url],
                check=True,
                capture_output=True,
                timeout=5,
            )
            print(f"Opened W&B URL in Chrome: {url}")
            return
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            continue

    print(f"Could not open Chrome. Please manually visit: {url}")


def stop_runpod(pod_id: str | None = None, api_key: str | None = None) -> bool:
    """Stop the active RunPod instance."""
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")

    if not pod_id or not api_key:
        print("RUNPOD_POD_ID or RUNPOD_API_KEY not set – skipping pod stop.")
        return False

    # Try the Python SDK first
    try:
        runpod.api_key = api_key
        if hasattr(runpod, "stop_pod"):
            runpod.stop_pod(pod_id)
            print("Successfully requested pod stop (SDK).")
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"SDK method failed: {exc}. Falling back to REST call...")

    # Fallback to direct REST API
    try:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print("Successfully requested pod stop (REST).")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to stop pod: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start training pod")
    t.add_argument("config", help="Training config file")
    t.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type name")
    t.add_argument("--api-key", help="RunPod API key")
    t.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep pod alive after training completes (disables auto-stop)",
    )
    t.add_argument("--note", help="Note to add to the W&B run")

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(
            args.config,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
            keep_alive=args.keep_alive,
            note=args.note,
        )
