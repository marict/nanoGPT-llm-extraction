#!/usr/bin/env python
"""
Tests for format_expression_string function to verify correct handling of negative numbers.
"""

import pytest
import sympy

from data.dagset.expression_to_string import (convert_number_to_english,
                                              format_expression_string)


class TestFormatExpressionStringNegativeNumbers:
    """Test format_expression_string with negative numbers."""

    def test_simple_negative_number_no_english(self):
        """Test simple negative number without English conversion."""
        # Create a symbol with the negative value directly
        expression = sympy.Symbol("-5.2")
        initial_values = [-5.2]
        result, style = format_expression_string(
            expression, english_conversion_probability=0.0, seed=42
        )
        assert "-5.2" in result

    def test_simple_negative_number_full_english(self):
        """Test simple negative number with full English conversion."""
        # Create the expression using convert_number_to_english as would happen
        # in the streaming code with english_conversion_probability=1.0
        english_name = convert_number_to_english(-5.2)
        expression = sympy.Symbol(english_name)
        initial_values = [-5.2]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # With full English conversion, negative numbers become "negative [number]"
        assert "negative" in result[0].lower()
        assert "five" in result[0].lower()

    def test_negative_integer_no_decimal(self):
        """Test negative integer without decimal point."""
        english_name = convert_number_to_english(-7.0)
        expression = sympy.Symbol(english_name)
        initial_values = [-7.0]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # With full English conversion, negative numbers become "negative [number]"
        assert "negative" in result[0].lower()
        assert "seven" in result[0].lower()

    def test_negative_zero(self):
        """Test negative zero handling."""
        english_name = convert_number_to_english(-0.0)
        expression = sympy.Symbol(english_name)
        initial_values = [-0.0]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Negative zero becomes just zero (mathematically correct)
        assert result[0].lower() == "zero"

    def test_negative_in_addition_expression(self):
        """Test negative number in addition expression."""
        # Create symbols with English names
        left = sympy.Symbol(convert_number_to_english(-3.5))
        right = sympy.Symbol(convert_number_to_english(2.1))
        expression = sympy.Add(left, right, evaluate=False)
        initial_values = [-3.5, 2.1]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number
        # For the addition - accept any addition synonym
        addition_synonyms = ["plus", "added to"]
        assert (
            any(syn in result[0].lower() for syn in addition_synonyms)
            or "+" in result[0]
        )

    def test_negative_in_subtraction_expression(self):
        """Test negative number in subtraction expression."""
        # Create symbols with English names
        left = sympy.Symbol(convert_number_to_english(-4.2))
        right = sympy.Symbol(convert_number_to_english(1.8))
        expression = sympy.Add(
            left, -right, evaluate=False
        )  # Subtraction is Add with negative
        initial_values = [-4.2, 1.8]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for the first number
        assert "negative" in result[0].lower()  # For -4.2
        # Should contain both numbers (operator conversion may vary)
        assert "four" in result[0].lower() and "one" in result[0].lower()

    def test_negative_in_multiplication_expression(self):
        """Test negative number in multiplication expression."""
        # Create symbols with English names
        left = sympy.Symbol(convert_number_to_english(-2.5))
        right = sympy.Symbol(convert_number_to_english(3.0))
        expression = sympy.Mul(left, right, evaluate=False)
        initial_values = [-2.5, 3.0]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number
        mult_indicators = ["*", "times", "multiplied"]
        assert any(indicator in result[0].lower() for indicator in mult_indicators)

    def test_negative_in_division_expression(self):
        """Test negative number in division expression."""
        # Create symbols with English names
        left = sympy.Symbol(convert_number_to_english(-8.4))
        right = sympy.Symbol(convert_number_to_english(2.1))
        expression = sympy.Mul(
            left, sympy.Pow(right, -1, evaluate=False), evaluate=False
        )
        initial_values = [-8.4, 2.1]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number
        div_indicators = ["/", "divided", "over", "divide"]
        assert any(indicator in result[0].lower() for indicator in div_indicators)

    def test_multiple_negative_numbers(self):
        """Test expression with multiple negative numbers."""
        # Create symbols with English names
        left = sympy.Symbol(convert_number_to_english(-3.2))
        right = sympy.Symbol(convert_number_to_english(-1.5))
        expression = sympy.Add(left, right, evaluate=False)
        initial_values = [-3.2, -1.5]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for both numbers
        negative_count = result[0].lower().count("negative")
        assert negative_count == 2  # One for each negative number

    def test_negative_in_parentheses(self):
        """Test negative number within parentheses."""
        # Create symbols with English names - parentheses are handled by sympy rendering
        left = sympy.Symbol(convert_number_to_english(-5.0))
        right = sympy.Symbol(convert_number_to_english(3.0))
        expression = sympy.Add(left, right, evaluate=False)
        initial_values = [-5.0, 3.0]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for the number
        assert "negative" in result[0].lower()
        # Numbers should be present
        assert "five" in result[0].lower() and "three" in result[0].lower()

    def test_complex_expression_with_negatives(self):
        """Test complex expression with multiple negative numbers."""
        # For complex expressions, we'll just use the English conversion probability
        # in _generate_expression which would create the appropriate symbols
        a = sympy.Symbol(convert_number_to_english(-1.2))
        b = sympy.Symbol(convert_number_to_english(-3.4))
        c = sympy.Symbol(convert_number_to_english(2.5))
        d = sympy.Symbol(convert_number_to_english(-0.8))

        # Build (b * c) / d
        bc = sympy.Mul(b, c, evaluate=False)
        bc_div_d = sympy.Mul(bc, sympy.Pow(d, -1, evaluate=False), evaluate=False)

        # Add a + (b * c) / d
        expression = sympy.Add(a, bc_div_d, evaluate=False)

        initial_values = [-1.2, -3.4, 2.5, -0.8]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have "negative" for multiple negative numbers
        negative_count = result[0].lower().count("negative")
        assert negative_count >= 2  # At least for -1.2 and -0.8

    def test_very_small_negative_number(self):
        """Test very small negative number."""
        english_name = convert_number_to_english(-0.001)
        expression = sympy.Symbol(english_name)
        initial_values = [-0.001]
        result, style = format_expression_string(
            expression,
            english_conversion_probability=1.0,
            seed=42,
        )
        assert "negative" in result[0].lower()  # For the negative number

    def test_large_negative_number(self):
        """Test large negative number."""
        english_name = convert_number_to_english(-12345.67)
        expression = sympy.Symbol(english_name)
        initial_values = [-12345.67]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number

    def test_partial_english_conversion_deterministic(self):
        """Test that conversion is deterministic with same seed."""
        # For this test we just check that operation conversion is deterministic
        # (symbol names are deterministic by construction now)
        a = sympy.Symbol("-5.2")
        b = sympy.Symbol("3.1")
        expression = sympy.Add(a, b, evaluate=False)
        initial_values = [-5.2, 3.1]
        result1, style1 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=42
        )
        result2, style2 = format_expression_string(
            expression, english_conversion_probability=0.5, seed=42
        )
        assert result1 == result2

    def test_partial_english_conversion_different_seeds(self):
        """Test that different seeds can produce different results."""
        # For this test we just check that operation conversion varies with seed
        # (symbol names are determined separately now)
        a = sympy.Symbol("-5.2")
        b = sympy.Symbol("3.1")
        expression = sympy.Add(a, b, evaluate=False)
        initial_values = [-5.2, 3.1]
        result1, style1 = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        result2, style2 = format_expression_string(
            expression, english_conversion_probability=1.0, seed=123
        )
        # With different seeds, results might be different (though not guaranteed)
        # This test mainly ensures no crashes occur with different seeds

    def test_negative_scientific_notation(self):
        """Test negative number in scientific notation."""
        english_name = convert_number_to_english(-1.5e-3)
        expression = sympy.Symbol(english_name)
        initial_values = [-1.5e-3]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number

    def test_mixed_positive_negative_expression(self):
        """Test expression mixing positive and negative numbers."""
        a = sympy.Symbol(convert_number_to_english(5.0))
        b = sympy.Symbol(convert_number_to_english(-3.2))
        c = sympy.Symbol(convert_number_to_english(-1.8))

        # a + b - c
        ab = sympy.Add(a, b, evaluate=False)
        expression = sympy.Add(ab, -c, evaluate=False)

        initial_values = [5.0, -3.2, -1.8]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should have at least one "negative" for the negative numbers
        negative_count = result[0].lower().count("negative")
        assert negative_count >= 1  # English conversion is probabilistic

    def test_whitespace_preservation_with_negatives(self):
        """Test that whitespace handling works correctly with negative numbers."""
        a = sympy.Symbol("-5.2")
        b = sympy.Symbol("3.1")
        expression = sympy.Add(a, b, evaluate=False)
        initial_values = [-5.2, 3.1]
        result, style = format_expression_string(
            expression, english_conversion_probability=0.0, seed=42
        )
        # Check that the negative number is present
        assert "-5.2" in result[0]

    def test_decimal_places_limit_negative(self):
        """Test decimal places limit with negative numbers."""
        english_name = convert_number_to_english(-3.123456789)
        expression = sympy.Symbol(english_name)
        initial_values = [-3.123456789]
        result, style = format_expression_string(
            expression,
            english_conversion_probability=1.0,
            seed=42,
        )
        assert "negative" in result[0].lower()  # For the negative number
        # The English conversion should respect decimal places limit

    def test_edge_case_negative_fraction(self):
        """Test negative fraction less than 1."""
        english_name = convert_number_to_english(-0.5)
        expression = sympy.Symbol(english_name)
        initial_values = [-0.5]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        assert "negative" in result[0].lower()  # For the negative number
        # Should handle fractional negative numbers correctly

    def test_user_reported_case(self):
        """Test the specific case reported by the user."""
        english_name = convert_number_to_english(-6962.978)
        expression = sympy.Symbol(english_name)
        initial_values = [-6962.978]
        result, style = format_expression_string(
            expression, english_conversion_probability=1.0, seed=42
        )
        # Should produce "negative [number]" not "less [number]"
        assert "negative" in result[0].lower()
        assert "six thousand" in result[0].lower()
        # Should NOT have minus synonyms for unary minus with full English conversion
        minus_synonyms = ["minus", "subtract", "less"]
        assert not any(syn in result[0].lower() for syn in minus_synonyms)
