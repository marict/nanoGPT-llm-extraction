#!/usr/bin/env python
"""
Tests for format_expression_string function to verify correct handling of negative numbers.
"""

import pytest

from data.dagset.expression_to_string import format_expression_string


class TestFormatExpressionStringNegativeNumbers:
    """Test format_expression_string with negative numbers."""

    def test_simple_negative_number_no_english(self):
        """Test simple negative number without English conversion."""
        expression = "-5.2"
        result = format_expression_string(
            expression, english_conversion_probability=0.0, seed=42
        )
        assert result == "- 5.2"  # Tokenizer treats - as separate operator

    def test_simple_negative_number_full_english(self):
        """Test simple negative number with full English conversion."""
        expression = "-5.2"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # With full English conversion, negative numbers become "negative [number]"
        assert "negative" in result.lower()
        assert "five" in result.lower()

    def test_negative_integer_no_decimal(self):
        """Test negative integer without decimal point."""
        expression = "-7"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # With full English conversion, negative numbers become "negative [number]"
        assert "negative" in result.lower()
        assert "seven" in result.lower()

    def test_negative_zero(self):
        """Test negative zero handling."""
        expression = "-0.0"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" and "zero"
        assert "negative" in result.lower() and "zero" in result.lower()

    def test_negative_in_addition_expression(self):
        """Test negative number in addition expression."""
        expression = "-3.5 + 2.1"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number
        assert "plus" in result.lower() or "+" in result  # For the addition

    def test_negative_in_subtraction_expression(self):
        """Test negative number in subtraction expression."""
        expression = "-4.2 - 1.8"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for the first number and "minus" for subtraction
        assert "negative" in result.lower()  # For -4.2
        assert "minus" in result.lower()  # For the subtraction operation

    def test_negative_in_multiplication_expression(self):
        """Test negative number in multiplication expression."""
        expression = "-2.5 * 3.0"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number
        mult_indicators = ["*", "times", "multiplied"]
        assert any(indicator in result.lower() for indicator in mult_indicators)

    def test_negative_in_division_expression(self):
        """Test negative number in division expression."""
        expression = "-8.4 / 2.1"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number
        div_indicators = ["/", "divided", "over", "divide"]
        assert any(indicator in result.lower() for indicator in div_indicators)

    def test_multiple_negative_numbers(self):
        """Test expression with multiple negative numbers."""
        expression = "-3.2 + -1.5"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for both numbers
        negative_count = result.lower().count("negative")
        assert negative_count == 2  # One for each negative number

    def test_negative_in_parentheses(self):
        """Test negative number within parentheses."""
        expression = "(-5.0) + 3.0"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for the number
        assert "negative" in result.lower()
        assert "(" in result and ")" in result

    def test_complex_expression_with_negatives(self):
        """Test complex expression with multiple negative numbers."""
        expression = "-1.2 + (-3.4 * 2.5) / -0.8"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for multiple negative numbers
        negative_count = result.lower().count("negative")
        assert negative_count >= 2  # At least for -1.2 and -0.8

    def test_very_small_negative_number(self):
        """Test very small negative number."""
        expression = "-0.001"
        result = format_expression_string(
            expression,
            english_conversion_probability=1.0,
            seed=42,
            max_decimal_places=6,
        )
        assert "negative" in result.lower()  # For the negative number

    def test_large_negative_number(self):
        """Test large negative number."""
        expression = "-12345.67"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number

    def test_partial_english_conversion_deterministic(self):
        """Test that conversion is deterministic with same seed."""
        expression = "-5.2 + 3.1"
        result1 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=42
        )
        result2 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=42
        )
        assert result1 == result2

    def test_partial_english_conversion_different_seeds(self):
        """Test that different seeds can produce different results."""
        expression = "-5.2 + 3.1"
        result1 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=42
        )
        result2 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=123
        )
        # With different seeds, results might be different (though not guaranteed)
        # This test mainly ensures no crashes occur with different seeds

    def test_negative_scientific_notation(self):
        """Test negative number in scientific notation."""
        expression = "-1.5e-3"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number

    def test_mixed_positive_negative_expression(self):
        """Test expression mixing positive and negative numbers."""
        expression = "5.0 + -3.2 - -1.8"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for the negative numbers
        negative_count = result.lower().count("negative")
        assert negative_count == 2  # For -3.2 and -1.8

    def test_whitespace_preservation_with_negatives(self):
        """Test that whitespace handling works correctly with negative numbers."""
        expression = " -5.2  +  3.1 "
        result = format_expression_string(
            expression, english_conversion_probability=0.0, seed=42
        )
        # Whitespace should be normalized and minus treated as separate token
        assert "- 5.2" in result
        assert "3.1" in result

    def test_decimal_places_limit_negative(self):
        """Test decimal places limit with negative numbers."""
        expression = "-3.123456789"
        result = format_expression_string(
            expression,
            english_conversion_probability=1.0,
            seed=42,
            max_decimal_places=3,
        )
        assert "negative" in result.lower()  # For the negative number
        # The English conversion should respect decimal places limit

    def test_edge_case_negative_fraction(self):
        """Test negative fraction less than 1."""
        expression = "-0.5"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result.lower()  # For the negative number
        # Should handle fractional negative numbers correctly

    def test_user_reported_case(self):
        """Test the specific case reported by the user."""
        expression = "-6962.978"
        result = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should produce "negative [number]" not "less [number]"
        assert "negative" in result.lower()
        assert "six thousand" in result.lower()
        # Should NOT have minus synonyms for unary minus with full English conversion
        minus_synonyms = ["minus", "subtract", "less"]
        assert not any(syn in result.lower() for syn in minus_synonyms)
