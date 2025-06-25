#!/usr/bin/env python3
"""
Tests for scripts/container_setup.sh

Run:  python -m pytest tests/test_container_setup.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

# project root = repo root two levels up from *this* file
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "container_setup.sh"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print_result(name: str, ok: bool) -> bool:
    icon = "✅" if ok else "❌"
    print(f"{icon} {name}")
    return ok


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #
def test_script_exists() -> None:
    assert _print_result("script exists", SCRIPT.exists())
    assert _print_result("script is executable", os.access(SCRIPT, os.X_OK))


def test_bash_syntax() -> None:
    print("\nchecking bash -n syntax …")
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    ok = proc.returncode == 0
    if not ok:
        print(proc.stderr.strip())
    assert _print_result("bash syntax valid", ok)


def test_content_tokens() -> None:
    required = {
        "shebang": {
            "token": "#!/usr/bin/env bash",
            "purpose": "Script interpreter declaration - tells system to run with bash",
        },
        "error handling": {
            "token": "set -e",
            "purpose": "Exit on any error - critical for container reliability",
        },
        "timing variable": {
            "token": "start_time=",
            "purpose": "Timing variable for log timestamps - needed for log() function",
        },
        "apt-get": {
            "token": "apt-get",
            "purpose": "System package installation - needed to install tree and other dependencies",
        },
        "pip install": {
            "token": "pip install",
            "purpose": "Python package installation - needed to install project dependencies",
        },
        "python train": {
            "token": "python -u train.py",
            "purpose": "Training script execution - core functionality of the container",
        },
        "arg forward": {
            "token": 'python -u train.py "$@"',
            "purpose": "Argument forwarding - allows passing training arguments to the script",
        },
        "training completion": {
            "token": "python -u train.py",
            "purpose": "Training script execution - container stops automatically via stop_runpod",
        },
        "log function": {
            "token": "log()",
            "purpose": "Logging function definition - provides structured logging with timestamps",
        },
        "cd to repo": {
            "token": "cd /workspace/repo",
            "purpose": "Change to repository directory - ensures we're in the right working directory",
        },
    }

    text = SCRIPT.read_text()
    print("\nscanning script tokens …")
    missing_tokens = []

    for label, info in required.items():
        token = info["token"]
        purpose = info["purpose"]
        found = token in text
        if not found:
            missing_tokens.append(f"  ❌ {label}: '{token}' - {purpose}")
        else:
            print(f"  ✅ {label}: '{token}'")

    if missing_tokens:
        print("\n❌ MISSING REQUIRED TOKENS:")
        for missing in missing_tokens:
            print(missing)
        print(f"\nThe container setup script is missing essential components.")
        print(f"Please check {SCRIPT} and ensure all required tokens are present.")
        assert (
            False
        ), f"Container setup script missing {len(missing_tokens)} required tokens"

    print(f"\n✅ All {len(required)} required tokens found in container setup script")


def test_structure_hints() -> None:
    lines = SCRIPT.read_text().splitlines()

    checks = [
        (
            "has shebang",
            lines[0].startswith("#!/usr/bin/env bash"),
            "First line must declare bash interpreter",
        ),
        (
            "has set -e",
            any("set -e" in l for l in lines),
            "Script must exit on errors for container reliability",
        ),
        (
            "has start_time var",
            any("start_time=" in l for l in lines),
            "Timing variable needed for log timestamps",
        ),
        (
            "has apt-get",
            any("apt-get" in l for l in lines),
            "System package installation required",
        ),
        (
            "has pip install",
            any("pip install" in l for l in lines),
            "Python package installation required",
        ),
        (
            "has training execution",
            any("python -u train.py" in l for l in lines),
            "Training script execution - container stops automatically",
        ),
        (
            "has log function",
            any("log()" in l for l in lines),
            "Logging function needed for structured output",
        ),
        (
            "has cd to repo",
            any("cd /workspace/repo" in l for l in lines),
            "Must change to repository directory",
        ),
    ]

    print("\nchecking structure hints …")
    missing_checks = []

    for label, ok, purpose in checks:
        if not ok:
            missing_checks.append(f"  ❌ {label}: {purpose}")
        else:
            print(f"  ✅ {label}")

    if missing_checks:
        print("\n❌ STRUCTURE ISSUES FOUND:")
        for missing in missing_checks:
            print(missing)
        print(f"\nThe container setup script has structural problems.")
        print(
            f"Please check {SCRIPT} and ensure all required structural elements are present."
        )
        assert (
            False
        ), f"Container setup script has {len(missing_checks)} structural issues"

    print(f"\n✅ All {len(checks)} structural checks passed")


# --------------------------------------------------------------------------- #
# main entry (optional standalone run)
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests: list[tuple[str, Callable[[], None]]] = [
        ("existence / exec", test_script_exists),
        ("bash syntax", test_bash_syntax),
        ("content tokens", test_content_tokens),
        ("structure", test_structure_hints),
    ]

    print("🧪 running container_setup.sh tests\n")
    try:
        for name, fn in tests:
            print(f"→ {name}")
            fn()
            print()
    except AssertionError:
        print("❌ some tests failed")
        return 1

    print("🎉 all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
