import os
import subprocess
import sys
from pathlib import Path


def test_train_predictor_default_args(tmp_path, monkeypatch):
    """Run ``train_predictor.py`` with its default configuration and ensure it exits
    successfully. The subprocess is executed in an isolated temporary working
    directory so that any checkpoints or artefacts are written outside the
    repository tree. WandB is forced into *offline* mode to avoid any network
    calls during CI runs.
    """

    # ------------------------------------------------------------------
    # Prepare environment variables
    # ------------------------------------------------------------------
    env = os.environ.copy()

    # Provide a dummy API key so that the script doesn't abort.
    env.setdefault("WANDB_API_KEY", "offline_dummy_key")

    # Force WandB into offline/dry-run mode to avoid network access.
    # Both variables are respected by WandB; setting both for robustness.
    env["WANDB_MODE"] = "dryrun"
    env["WANDB_SILENT"] = "true"

    # Ensure the repository root is on PYTHONPATH so that ``train_predictor``
    # can import local modules even though we run in a temporary directory.
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        f"{repo_root}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(repo_root)
    )

    # ------------------------------------------------------------------
    # Build the command: run the script *without* extra arguments so it picks
    # up the default minimal config (``config/train_predictor_default.py``).
    # We only need to pass a dummy W&B key explicitly to placate the parser.
    # ------------------------------------------------------------------
    script_path = repo_root / "train_predictor.py"
    # The script expects the configuration file path as the first positional
    # argument *unless* it can find the default relative path. Because we run
    # in a temp directory, we must pass the absolute path explicitly.
    cfg_path = repo_root / "config" / "train_predictor_default.py"
    cmd = [
        sys.executable,
        str(script_path),
        str(cfg_path),
        "--wandb-api-key",
        "offline_dummy_key",
    ]

    # Run inside the temporary directory to isolate artefacts (checkpoints,
    # WandB files, etc.). Capture output for debugging and enforce a timeout
    # so that CI doesn't hang in the unlikely event of an issue.
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,  # 3 minutes should be plenty even on slow CI machines
    )

    # Print captured output on failure for easier debugging
    if result.returncode != 0:
        print(result.stdout)

    # The script currently swallows exceptions and exits with status 0 while
    # printing a "Fatal error" message. Guard against that so the test fails
    # when the training loop encounters an exception.
    error_indicators = [
        "Fatal error in training loop",
        "Fatal error in predictor training",
        "Traceback (most recent call last)",
        "RuntimeError",
        "AttributeError",
    ]

    error_found = any(indicator in result.stdout for indicator in error_indicators)

    assert result.returncode == 0 and not error_found, (
        "train_predictor.py exited with errors. Output:\n" + result.stdout
    )
