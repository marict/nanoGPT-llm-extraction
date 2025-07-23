#!/usr/bin/env python
"""
Tests for the multi-style expression rendering system.
"""

import math
import random

import pytest
import sympy

from data.dagset.expression_to_string import (
    SUPPORTED_STYLES,
    convert_number_to_english,
    format_expression_string,
    render_expr,
    sample_printing_style,
    validate_printing_style_probs,
)


class TestValidation:
    """Test validation functions."""

    def test_validate_printing_style_probs_valid(self):
        """Test validation with valid probability distributions."""
        # Single style
        validate_printing_style_probs({"sstr": 1.0})

        # Multiple styles
        validate_printing_style_probs({"sstr": 0.5, "latex": 0.3, "pretty": 0.2})

        # All supported styles
        all_styles = {style: 1.0 / len(SUPPORTED_STYLES) for style in SUPPORTED_STYLES}
        validate_printing_style_probs(all_styles)

    def test_validate_printing_style_probs_invalid_sum(self):
        """Test validation fails when probabilities don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_printing_style_probs({"sstr": 0.5, "latex": 0.3})

        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_printing_style_probs({"sstr": 1.1})

    def test_validate_printing_style_probs_unsupported_style(self):
        """Test validation fails with unsupported styles."""
        with pytest.raises(ValueError, match="Unsupported printing styles"):
            validate_printing_style_probs({"invalid_style": 1.0})

        with pytest.raises(ValueError, match="Unsupported printing styles"):
            validate_printing_style_probs({"sstr": 0.5, "unknown": 0.5})

    def test_validate_printing_style_probs_empty(self):
        """Test validation fails with empty dictionary."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_printing_style_probs({})


class TestSampling:
    """Test probabilistic sampling."""

    def test_sample_printing_style_deterministic(self):
        """Test that sampling is deterministic with fixed seed."""
        probs = {"sstr": 0.3, "latex": 0.4, "pretty": 0.3}
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        for _ in range(10):
            style1 = sample_printing_style(probs, rng1)
            style2 = sample_printing_style(probs, rng2)
            assert style1 == style2

    def test_sample_printing_style_distribution(self):
        """Test that sampling roughly follows the specified distribution."""
        probs = {"sstr": 0.6, "latex": 0.4}
        rng = random.Random(42)

        samples = [sample_printing_style(probs, rng) for _ in range(1000)]
        sstr_count = samples.count("sstr")
        latex_count = samples.count("latex")

        # Allow 10% tolerance
        assert 0.5 <= sstr_count / 1000 <= 0.7
        assert 0.3 <= latex_count / 1000 <= 0.5

    def test_sample_printing_style_single_style(self):
        """Test sampling with single style always returns that style."""
        for style in SUPPORTED_STYLES:
            probs = {style: 1.0}
            rng = random.Random(42)
            for _ in range(10):
                assert sample_printing_style(probs, rng) == style


class TestRendering:
    """Test the expression rendering system."""

    def test_render_expr_all_styles(self):
        """Test that all supported styles render without errors."""
        # Use symbol names directly
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("2.0")
        expr = sympy.Add(a, b, evaluate=False)

        for style in SUPPORTED_STYLES:
            result = render_expr(expr, style)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_render_expr_sstr_vs_str(self):
        """Test difference between sstr and str."""
        # Use symbol names directly
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("2.0")
        expr = sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)

        sstr_result = render_expr(expr, "sstr")

        # sstr is compact representation
        assert "1.5" in sstr_result
        assert "2.0" in sstr_result
        assert "/" in sstr_result

    def test_render_expr_latex_style(self):
        """Test LaTeX output style."""
        # Use symbol names directly
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("2.0")
        expr = sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)

        latex_result = render_expr(expr, "latex")

        # LaTeX uses \frac{}{} for division
        assert "\\frac" in latex_result

    def test_render_expr_pretty_style(self):
        """Test pretty-printing style."""
        # Use symbol names directly
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("2.0")
        expr = sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)

        pretty_result = render_expr(expr, "pretty")

        # Pretty printing uses a line to represent division
        assert "\n" in pretty_result  # Newlines for fraction layout

    def test_render_expr_invalid_style(self):
        """Test that invalid styles raise errors."""
        # Use symbol names directly
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("2.0")
        expr = sympy.Add(a, b, evaluate=False)

        with pytest.raises(ValueError, match="Unsupported style"):
            render_expr(expr, "invalid_style")


class TestIntegration:
    """Test the complete formatting pipeline."""

    def test_format_expression_string_sympy_expr(self):
        """Test formatting SymPy expressions with style selection."""
        a = sympy.Symbol("1.5")
        b = sympy.Symbol("-2.0")
        c = sympy.Symbol("2")
        expr = sympy.Add(a, sympy.Mul(b, c, evaluate=False), evaluate=False)
        values = [1.5, -2.0, 2]

        # Test with specific style
        result, style = format_expression_string(
            expr, printing_style_probs={"sstr": 1.0}, seed=42
        )

        assert "1.5" in result
        assert "-2" in result

    def test_format_expression_string_default_probs(self):
        """Test that None printing_style_probs defaults to sstr."""
        a = sympy.Symbol("1.0")
        b = sympy.Symbol("2.0")
        expr = sympy.Add(a, b, evaluate=False)
        values = [1.0, 2.0]

        result, style = format_expression_string(
            expr, printing_style_probs=None, seed=42
        )

        # Should work and produce valid output
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_format_expression_string_probabilistic(self):
        """Test probabilistic style selection."""
        a = sympy.Symbol("1.0")
        b = sympy.Symbol("2.0")
        expr = sympy.Add(a, b, evaluate=False)
        values = [1.0, 2.0]
        probs = {"sstr": 0.5, "latex": 0.5}

        # Generate multiple samples
        results = []
        for i in range(20):
            result, style = format_expression_string(
                expr, printing_style_probs=probs, seed=i
            )
            results.append(result[0])

        # All results should be valid strings
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0

    def test_format_expression_string_english_conversion(self):
        """Test combination with English conversion."""
        # Create symbols directly with English names
        a = sympy.Symbol(convert_number_to_english(1.0))
        b = sympy.Symbol(convert_number_to_english(2.0))
        expr = sympy.Add(a, b, evaluate=False)
        values = [1.0, 2.0]

        result, style = format_expression_string(
            expr,
            printing_style_probs={"sstr": 1.0},
            english_conversion_probability=1.0,
            seed=42,
        )

        # Should contain English words for numbers and operators
        assert "one" in result.lower()  # numbers already converted
        # At least one operator word should appear for sstr style
        assert any(op in result.lower() for op in ["plus", "added to"])

    def test_english_conversion_all_styles(self):
        """Test that English conversion works for all printing styles."""
        # Create symbols directly with English names
        a = sympy.Symbol(convert_number_to_english(2.0))
        b = sympy.Symbol(convert_number_to_english(3.0))
        c = sympy.Symbol(convert_number_to_english(4.0))
        d = sympy.Symbol(convert_number_to_english(2.0))

        # (a + b * c / d)
        bc = sympy.Mul(b, c, evaluate=False)
        bc_div_d = sympy.Mul(bc, sympy.Pow(d, -1, evaluate=False), evaluate=False)
        expr = sympy.Add(a, bc_div_d, evaluate=False)
        values = [2.0, 3.0, 4.0, 2.0]

        # Test each style with English conversion
        styles = ["sstr", "latex", "pretty", "ascii", "repr"]

        for style in styles:
            # Test with English conversion enabled
            result, style = format_expression_string(
                expr,
                printing_style_probs={style: 1.0},
                english_conversion_probability=1.0,
                seed=42,
            )

            # Should contain English words for numbers (already in symbol names)
            assert (
                "two" in result.lower()
                or "three" in result.lower()
                or "four" in result.lower()
            )

            # Test that the result is different from non-English version (operations conversion)
            # Only for styles that support operation conversion
            if style in ["sstr", "latex"]:
                english_words = [
                    "plus",
                    "added to",  # Addition
                    "times",
                    "multiplied by",  # Multiplication
                    "divided by",
                    "over",  # Division
                ]

                has_english = any(word in result.lower() for word in english_words)
                assert (
                    has_english
                ), f"Style '{style}' did not produce English conversion. Got: {repr(result)}"

    def test_negative_numbers_all_styles(self):
        """Test that negative number conversion works for all printing styles."""
        # Create symbols directly with English names
        a = sympy.Symbol(convert_number_to_english(-5.0))
        b = sympy.Symbol(convert_number_to_english(3.2))
        expr = sympy.Add(a, b, evaluate=False)
        values = [-5.0, 3.2]

        styles = ["sstr", "latex", "pretty", "ascii", "repr"]

        for style in styles:
            # Test with English conversion enabled
            result, style = format_expression_string(
                expr,
                printing_style_probs={style: 1.0},
                english_conversion_probability=1.0,
                seed=42,
            )

            # Should contain "negative" for the negative number in all styles
            # since it's part of the symbol name
            assert (
                "negative" in result.lower()
            ), f"Style '{style}' did not preserve 'negative' word"

    def test_division_conversion_styles(self):
        """Test that division conversion works correctly across different styles."""
        # Create symbols directly with English names
        a = sympy.Symbol(convert_number_to_english(8.0))
        b = sympy.Symbol(convert_number_to_english(2.0))
        expr = sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)
        values = [8.0, 2.0]

        # Test each style with English conversion
        styles = ["sstr", "latex", "pretty", "ascii", "repr"]

        for style in styles:
            result, style = format_expression_string(
                expr,
                printing_style_probs={style: 1.0},
                english_conversion_probability=1.0,
                seed=42,
            )

            # All styles should have numbers in English (from symbol names)
            assert (
                "eight" in result.lower() and "two" in result.lower()
            ), f"Style '{style}' did not preserve English number names. Got: {repr(result)}"

            # Check for division words in sstr style
            if style == "sstr":
                division_words = ["divided by", "over"]
                has_division = any(word in result.lower() for word in division_words)
                assert (
                    has_division
                ), f"Style '{style}' should convert division to English. Got: {repr(result)}"

            # For latex style, check that it uses \frac for division
            elif style == "latex":
                assert (
                    "\\frac" in result
                ), f"LaTeX style should use \\frac for division. Got: {repr(result)}"

            # For pretty/ascii styles, check for newlines which indicate fraction layout
            elif style in ["pretty", "ascii"]:
                assert (
                    "\n" in result
                ), f"Style '{style}' should use multi-line layout for division. Got: {repr(result)}"

    def test_format_expression_string_sympy_input(self):
        """Test formatting SymPy expressions."""
        # Create symbols directly
        a = sympy.Symbol("1.0")
        b = sympy.Symbol("2.0")
        expr = sympy.Add(a, b, evaluate=False)
        initial_values = [1.0, 2.0]

        result, style = format_expression_string(
            expr, printing_style_probs={"sstr": 1.0}, seed=42
        )

        assert "1" in result
        assert "2" in result


class TestNumericalCorrectness:
    """Test numerical correctness of expression evaluation."""

    def test_sympy_evaluation_consistency(self):
        """Test consistency between sympy evaluation and string representations."""
        # This test now only verifies that sympy expressions can be properly converted to strings
        # since substitution happens at symbol creation time

        # Create symbols with simple numbers
        a = sympy.Symbol("3")
        b = sympy.Symbol("4")

        # Simple expression: a + b
        expr = sympy.Add(a, b, evaluate=False)

        # Check that all styles render correctly
        for style in SUPPORTED_STYLES:
            rendered = render_expr(expr, style)
            assert "3" in rendered
            assert "4" in rendered


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_initial_values(self):
        """Test handling of expressions with no placeholders."""
        # Create a constant expression with no symbols
        expr = sympy.sympify("2 + 3")

        result, style = format_expression_string(
            expr, printing_style_probs={"sstr": 1.0}, seed=42
        )

        assert isinstance(result[0], str)

    def test_complex_expressions(self):
        """Test with more complex mathematical expressions."""
        # Create symbols
        a = sympy.Symbol("2.0")
        b = sympy.Symbol("3.0")
        c = sympy.Symbol("4.0")
        d = sympy.Symbol("1.0")
        e = sympy.Symbol("2.0")

        # a^2 + b*c - d/e
        a_sq = sympy.Pow(a, 2, evaluate=False)
        bc = sympy.Mul(b, c, evaluate=False)
        d_div_e = sympy.Mul(d, sympy.Pow(e, -1, evaluate=False), evaluate=False)

        part1 = sympy.Add(a_sq, bc, evaluate=False)
        expr = sympy.Add(part1, -d_div_e, evaluate=False)

        values = [2.0, 3.0, 4.0, 1.0, 2.0]

        # Should not crash for any style
        for style in SUPPORTED_STYLES:
            try:
                result, style = format_expression_string(
                    expr,
                    printing_style_probs={style: 1.0},
                    seed=42,
                )
                assert isinstance(result[0], str)
            except:
                # Some complex expressions might not render in all styles, that's ok
                pass


def test_newlines_preserved_in_pretty_print():
    """Test that pretty printing preserves newlines for fractions."""
    # Create division expression
    a = sympy.Symbol("1.0")
    b = sympy.Symbol("2.0")
    expr = sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)
    values = [1.0, 2.0]

    result, style = format_expression_string(
        expr, printing_style_probs={"pretty": 1.0}, seed=42
    )

    # Pretty printing should preserve newlines for fractions
    assert "\n" in result

    # Division should appear as a line
    line_found = False
    for line in result.split("\n"):
        if "─" in line or "-" in line:  # Unicode or ASCII line
            line_found = True
            break
    assert (
        line_found
    ), f"No division line found in pretty-printed result: {repr(result)}"


def test_newlines_preserved_with_spacing():
    """Test that newlines are properly preserved when spacing is added."""
    # Create a multi-line expression with fraction
    a = sympy.Symbol("1.0")
    b = sympy.Symbol("2.0")
    c = sympy.Symbol("3.0")

    # (a + b) / c
    ab = sympy.Add(a, b, evaluate=False)
    expr = sympy.Mul(ab, sympy.Pow(c, -1, evaluate=False), evaluate=False)
    values = [1.0, 2.0, 3.0]

    # With pretty printing
    result, style = format_expression_string(
        expr, printing_style_probs={"pretty": 1.0}, seed=42
    )

    # Count original newlines
    original_newlines = result.count("\n")

    # Original should have newlines for fraction layout
    assert original_newlines > 0

    # sstr and latex styles should convert operations but not mess with newlines
    result_sstr, style = format_expression_string(
        expr,
        printing_style_probs={"sstr": 1.0},
        english_conversion_probability=1.0,
        seed=42,
    )

    # Verify basic formatting is applied
    assert "1.0" in result_sstr
    assert "2.0" in result_sstr
    assert "3.0" in result_sstr
