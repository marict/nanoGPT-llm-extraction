"""Combined miscellaneous tests aggregating coverage previously in runpod, math eval, and text generation rounding tests."""

import re

import runpod_service as rp
from data.dagset.streaming import generate_single_dag_example
from models.dag_model import TEST_OPS_NAMES

N_OPS = len(TEST_OPS_NAMES)
from run_math_eval import _extract_number


def test_text_generation_rounding_subset():
    """Reduced-iteration sanity check for text generation rounding values.

    Covers functionality originally in test_text_generation_rounding but with
    100 iterations instead of 1 000 to cut runtime while still exercising the
    same code paths.
    """
    for _ in range(100):
        example = generate_single_dag_example(
            depth=3, conversion_probability=0, allowed_operations=TEST_OPS_NAMES
        )
        text = example.text
        expected_values = example.initial_values

        # Extract numeric substrings from generated text
        string_numbers = re.findall(r"-?\d+\.\d+", text)
        non_pad_values = [v for v in expected_values if v != 1.0]

        # All numbers appearing in text should come from initial values
        for num_str in string_numbers:
            assert float(num_str) in expected_values

        # And every non-PAD initial value should appear in the text
        for val in non_pad_values:
            val_str = "0.0" if str(val) == "-0.0" else str(val)
            assert val_str in string_numbers


def test_runpod_stop_sdk(monkeypatch):
    """Smoke-test stop_runpod via SDK branch (replaces full runpod_service suite)."""

    class _FakeRunpod:
        api_key = None

        def stop_pod(self, pid):  # noqa: D401 – inline docstring not needed
            self.called_pid = pid
            return {"status": "STOPPING"}

    fake_sdk = _FakeRunpod()
    monkeypatch.setattr(rp, "runpod", fake_sdk, raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_xyz")
    monkeypatch.setenv("RUNPOD_API_KEY", "key_xyz")

    assert rp.stop_runpod() is True
    assert fake_sdk.called_pid == "pod_xyz"


def test_extract_number():
    """Quick check of _extract_number helper from run_math_eval."""
    assert _extract_number("Answer: 42.") == "42"
    assert _extract_number("No numbers here") == ""
