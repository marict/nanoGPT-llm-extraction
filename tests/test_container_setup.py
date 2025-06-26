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


def test_container_setup_comprehensive():
    """Comprehensive test of container setup script existence, syntax, and required content."""

    # Test 1: Basic file checks
    assert SCRIPT.exists(), f"Container setup script not found at {SCRIPT}"
    assert os.access(
        SCRIPT, os.X_OK
    ), f"Container setup script is not executable: {SCRIPT}"

    # Test 2: Bash syntax validation
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert (
        proc.returncode == 0
    ), f"Bash syntax error in {SCRIPT}:\n{proc.stderr.strip()}"

    # Test 3: Required content and structure
    text = SCRIPT.read_text()
    lines = text.splitlines()

    # Required tokens with their purposes
    required_checks = [
        (
            "#!/usr/bin/env bash",
            "Script interpreter declaration",
            lambda: lines[0].startswith("#!/usr/bin/env bash"),
        ),
        (
            "set -e",
            "Exit on any error for container reliability",
            lambda: "set -e" in text,
        ),
        (
            "start_time=",
            "Timing variable for log timestamps",
            lambda: "start_time=" in text,
        ),
        ("apt-get", "System package installation", lambda: "apt-get" in text),
        ("pip install", "Python package installation", lambda: "pip install" in text),
        (
            "python -u train.py",
            "Training script execution",
            lambda: "python -u train.py" in text,
        ),
        (
            'python -u train.py "$@"',
            "Argument forwarding capability",
            lambda: 'python -u train.py "$@"' in text,
        ),
        (
            "tail -f /dev/null",
            "Keep-alive support",
            lambda: "tail -f /dev/null" in text,
        ),
        ("log()", "Logging function definition", lambda: "log()" in text),
        (
            "cd /workspace/repo",
            "Change to repository directory",
            lambda: "cd /workspace/repo" in text,
        ),
    ]

    # Check all requirements
    missing_requirements = []
    for token, purpose, check_fn in required_checks:
        if not check_fn():
            missing_requirements.append(f"Missing {token}: {purpose}")

    assert not missing_requirements, (
        f"Container setup script missing {len(missing_requirements)} required elements:\n"
        + "\n".join(f"  - {req}" for req in missing_requirements)
    )


# --------------------------------------------------------------------------- #
# main entry (optional standalone run)
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    print("🧪 running container_setup.sh tests\n")
    try:
        test_container_setup_comprehensive()
        print("🎉 all tests passed")
        return 0
    except AssertionError as e:
        print(f"❌ test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(_run_all())
