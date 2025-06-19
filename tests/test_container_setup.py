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
        "shebang": "#!/bin/bash",
        "error handling": "set -e",
        "timing variable": "start_time=",
        "git clone": "git clone",
        "apt-get": "apt-get",
        "pip install": "pip install",
        "python train": "python train.py",
        "arg forward": 'python train.py "$@"',
        "tail keep-alive": "tail -f /dev/null",
    }

    text = SCRIPT.read_text()
    print("\nscanning script tokens …")
    for label, token in required.items():
        assert _print_result(label, token in text)


def test_structure_hints() -> None:
    lines = SCRIPT.read_text().splitlines()

    checks: list[tuple[str, bool]] = [
        ("has shebang", lines[0].startswith("#!/bin/bash")),
        ("has set -e", any("set -e" in l for l in lines)),
        ("has start_time var", any("start_time=" in l for l in lines)),
        ("has git ops", any("git" in l for l in lines)),
        ("has apt-get", any("apt-get" in l for l in lines)),
        ("has pip install", any("pip install" in l for l in lines)),
        ("has python train.py", any("python train.py" in l for l in lines)),
        ("has tail keep-alive", any("tail -f /dev/null" in l for l in lines)),
    ]

    print("\nchecking structure hints …")
    for label, ok in checks:
        assert _print_result(label, ok)


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
