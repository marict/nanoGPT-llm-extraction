import argparse
import os
import subprocess
import time
from typing import Sequence

import requests
import runpod

import wandb
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
# DEFAULT_GPU_TYPE = "NVIDIA A100-SXM4-80GB"
REPO_URL = "https://github.com/marict/nanoGPT-llm-extraction.git"


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
    keep_alive: bool = False,
    note: str | None = None,
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
    pod_name = "daggpt-train"  # default pod name
    project_name = "daggpt-train"  # default wandb project name
    if args_list:
        cfg_path = args_list[0]
        if not os.path.isabs(cfg_path):
            args_list[0] = f"{cfg_path}"
        # Extract the name from config for pod naming and wandb project
        try:
            data: dict[str, str] = {}
            with open(cfg_path, "r") as f:
                exec(f.read(), data)
            pod_name = data.get("name", "daggpt-train")
            project_name = data.get("name", "daggpt-train")
        except Exception:
            pass
    # Append optional note to the derived names so that runs can be easily identified
    if note:
        pod_name = f"{pod_name} - {note}"
        project_name = f"{project_name} - {note}"
    train_args = " ".join(args_list)

    gpu_type_id = _resolve_gpu_id(gpu_type)

    # Initialize wandb locally first to get the run ID
    wandb_result = init_local_wandb_and_open_browser(project_name, pod_name)

    # Docker args preparation
    docker_start = time.time()

    # Create an inline script that clones the repo first, then runs the setup script
    # Add keep-alive flag to training args if requested
    final_train_args = train_args
    if keep_alive:
        final_train_args += " --keep-alive"

    # Pass note through to remote train script so cfg.name matches and wandb resumes correctly
    if note:
        # Quote the note to preserve spaces if any
        escaped_note = note.replace('"', '\\"')
        final_train_args += f' --note="{escaped_note}"'

    # Add wandb run ID if available
    if wandb_result:
        wandb_url, wandb_run_id = wandb_result
        print(f"Wandb project initialized locally. URL: {wandb_url}")
        # Pass the wandb run ID to the training script so it can resume the same run
        final_train_args += f" --wandb-run-id={wandb_run_id}"

    inline_script = (
        f"apt-get update && apt-get install -y git && "
        f"cd /workspace && "
        f"( [ -d repo/.git ] && git -C repo pull || git clone https://github.com/marict/nanoGPT-llm-extraction.git repo ) && "
        f"bash /workspace/repo/scripts/container_setup.sh {final_train_args}"
    )
    docker_args = f"bash -c '{inline_script}'"
    print(
        f"[{time.time() - docker_start:.2f}s] Docker args preparation completed in {time.time() - docker_start:.2f}s"
    )
    pod = runpod.create_pod(
        name=pod_name,
        image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        min_vcpu_count=8,
        min_memory_in_gb=64,
        volume_in_gb=1000,  # persists across stops
        container_disk_in_gb=1000,  # wiped on stop
        network_volume_id="tvi2olc54y",
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

    # ------------------------------------------------------------------ #
    #  Update W&B run name to the pod id (plus optional note) so that
    #  the run name matches the actual instance identifier. This occurs
    #  *after* pod creation because we need the id first. The remote
    #  training job will resume the run by its id, so changing the name
    #  here is safe and ensures consistent naming across local/remote.
    # ------------------------------------------------------------------ #
    try:
        if wandb_result:
            run = wandb.run  # The current run initialised earlier
            if run is not None:
                new_run_name = pod_id if not note else f"{pod_id} - {note}"
                run.name = new_run_name
                run.save()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Failed to update wandb run name: {exc}")

    print(f"Starting training job '{pod_name}' (pod {pod_id}) on {gpu_type}. ")

    return pod_id


def init_local_wandb_and_open_browser(
    project_name: str, run_name: str
) -> tuple[str, str] | None:
    """
    Initialize wandb project locally and open the run URL in Chrome.

    Args:
        project_name: Name of the wandb project
        run_name: Run name (typically the pod name)

    Returns:
        Tuple of (wandb_url, run_id) if successful, None otherwise
    """
    try:
        # Check if WANDB_API_KEY is available
        if not os.getenv("WANDB_API_KEY"):
            print("Warning: WANDB_API_KEY not set, skipping local wandb initialization")
            return None

        # Initialize wandb run
        run = wandb.init(
            project=project_name,
            name=run_name,
            tags=["runpod", "remote-training"],
            notes=f"Remote training on RunPod instance {run_name}",
        )

        wandb_url = run.url + "/logs"  # Open directly to logs page
        wandb_run_id = run.id
        print(f"Wandb run created: {wandb_url}")

        # Try to open in Chrome
        try:
            # Try different Chrome executable names based on platform
            chrome_commands = [
                "google-chrome",  # Linux
                "google-chrome-stable",  # Linux alternative
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
                "chrome",  # Windows/generic
                "chromium",  # Linux alternative
                "chromium-browser",  # Linux alternative
            ]

            opened = False
            for chrome_cmd in chrome_commands:
                try:
                    subprocess.run(
                        [chrome_cmd, wandb_url],
                        check=True,
                        capture_output=True,
                        timeout=5,
                    )
                    print(f"Opened wandb URL in Chrome: {wandb_url}")
                    opened = True
                    break
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    continue

            if not opened:
                print(f"Could not open Chrome. Please manually visit: {wandb_url}")

        except Exception as e:
            print(f"Failed to open browser: {e}")
            print(f"Please manually visit: {wandb_url}")

        return (wandb_url, wandb_run_id)

    except Exception as e:
        print(f"Failed to initialize wandb locally: {e}")
        return None


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
    t.add_argument(
        "--note",
        help="Note to add to the wandb run",
    )

    args = parser.parse_args()
    if args.cmd == "train":
        start_cloud_training(
            args.config,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
            keep_alive=args.keep_alive,
            note=args.note,
        )
