import re
import textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_nvidia_cuda_lines(text: str) -> str:
    """Replicate the sed logic `/nvidia/Id;/cuda/Id` in pure Python.

    Any line containing the substrings "nvidia" or "cuda" (case-insensitive)
    is removed. The remaining lines are returned joined by newlines, preserving
    original order.
    """

    pattern = re.compile(r"(nvidia|cuda)", flags=re.IGNORECASE)
    kept_lines = [line for line in text.splitlines() if not pattern.search(line)]
    return "\n".join(kept_lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strip_nvidia_cuda_lines():
    """The stripping helper should drop any lines referencing NVIDIA or CUDA."""

    sample = textwrap.dedent(
        """\
        deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /
        deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
        deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
        deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /
        """
    )

    cleaned = _strip_nvidia_cuda_lines(sample)

    # Ensure references are gone and that the safe Ubuntu mirrors remain.
    assert "nvidia" not in cleaned.lower()
    assert "cuda" not in cleaned.lower()
    assert "archive.ubuntu.com" in cleaned
    assert "security.ubuntu.com" in cleaned
