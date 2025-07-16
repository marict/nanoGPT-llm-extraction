import re

import pytest

from data.dagset.streaming import plan_to_string_expression


class TestNumberFormatting:
    """Validate that numeric literals in generated expressions do not include
    excessively long binary-float artifacts such as 2.7800000000000002.
    """

    @pytest.mark.parametrize(
        "initial_values,operations",
        [
            (
                [0.14322, -2.7800000000000002, -0.651, 1.0, 1.0],
                ["add", "add", "identity", "identity"],
            ),
            (
                [5.0, 3.0],
                ["add"],
            ),
        ],
    )
    def test_no_long_repr_and_correct_trimming(self, initial_values, operations):
        expr = plan_to_string_expression(
            initial_values=initial_values,
            operations=operations,
            seed=123,
            english_conversion_probability=0.0,
        )

        # Expression should not contain long repeating zero artefacts.
        assert "000000000000" not in expr, f"Unexpected long float repr in: {expr}"
        assert "2.7800000000000002" not in expr, f"Untrimmed repr in: {expr}"

        # If the problematic value exists among inputs, ensure the trimmed version is present.
        if any(abs(v + 2.78) < 1e-12 for v in initial_values):
            assert (
                "-2.78" in expr or "2.78" in expr
            ), f"Trimmed value missing in: {expr}"

        # Integers should still carry a `.0` suffix according to formatting rules.
        for v in initial_values:
            if float(v).is_integer():
                # Only check formatting if this value actually appears in the expression.
                int_pattern = rf"\b{int(v)}\.0\b"
                presence_pattern = rf"\b{int(v)}(\.0)?\b"
                if re.search(presence_pattern, expr):
                    assert re.search(
                        int_pattern, expr
                    ), f"Integer {v} should appear as '{int(v)}.0' in expression: {expr}"
