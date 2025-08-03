"""
Test the core simplifications in normalize_expression:
1. Division: Mul(a, Pow(b, -1)) -> Div(a, b)
2. Subtraction: Add(a, Mul(-1, b)) -> Sub(a, b)
"""

import sympy

from data.dagset.streaming import Div, Sub, normalize_expression


def test_division_normalization():
    """Test that SymPy's division representation gets normalized to Div."""

    # Case 1: Simple division a/b
    expr = sympy.parse_expr("2.5/3.0", evaluate=False)
    normalized = normalize_expression(expr)

    print(f"Division test: {expr} -> {normalized}")
    assert isinstance(normalized, Div)
    assert str(normalized.args[0]) == "2.5"
    assert str(normalized.args[1]) == "3.0"

    # Case 2: Complex division
    expr = sympy.parse_expr("(a + b)/c", evaluate=False)
    normalized = normalize_expression(expr)

    print(f"Complex division: {expr} -> {normalized}")
    assert isinstance(normalized, Div)


def test_subtraction_normalization():
    """Test that SymPy's subtraction representation gets normalized to Sub."""

    # Case 1: Simple subtraction a - b (represented as a + (-1)*b)
    a = sympy.Symbol("a")
    b = sympy.Symbol("b")
    expr = a + sympy.Mul(-1, b, evaluate=False)  # a + (-1)*b

    normalized = normalize_expression(expr)
    print(f"Subtraction test: {expr} -> {normalized}")

    assert isinstance(normalized, Sub)
    assert normalized.args[0] == a
    assert normalized.args[1] == b


def test_problematic_expression():
    """Test the specific expression that was collapsing."""

    # The problematic inner expression: -1*770320.0 - 702533.714
    # This should become: Sub(Mul(-1, 770320.0), 702533.714)
    expr_str = "-1*770320.0 - 702533.714"
    expr = sympy.parse_expr(expr_str, evaluate=False)

    print(f"Original: {expr}")
    print(f"Structure: {sympy.srepr(expr)}")

    normalized = normalize_expression(expr)
    print(f"Normalized: {normalized}")
    print(f"Type: {type(normalized)}")

    # Should be Sub operation, not a collapsed float
    assert isinstance(normalized, Sub)

    # The full problematic expression
    full_expr_str = "-267.4*(-1*770320.0 - 702533.714)"
    full_expr = sympy.parse_expr(full_expr_str, evaluate=False)

    print(f"\\nFull expression: {full_expr}")
    normalized_full = normalize_expression(full_expr)
    print(f"Normalized full: {normalized_full}")

    # Should preserve structure, not collapse to a single float
    assert not isinstance(normalized_full, (sympy.Float, sympy.Integer))


def test_mixed_operations():
    """Test expressions with both division and subtraction."""

    # Expression with both: (a - b) / c
    expr = sympy.parse_expr("(a - b)/c", evaluate=False)
    normalized = normalize_expression(expr)

    print(f"Mixed operations: {expr} -> {normalized}")

    # Should be Div(Sub(a, b), c)
    assert isinstance(normalized, Div)
    assert isinstance(normalized.args[0], Sub)


def test_no_unwanted_changes():
    """Test that expressions without div/sub patterns are left alone."""

    # Simple addition - should stay as Add
    expr = sympy.parse_expr("a + b", evaluate=False)
    normalized = normalize_expression(expr)

    print(f"Simple addition: {expr} -> {normalized}")
    assert isinstance(normalized, sympy.Add)

    # Simple multiplication - should stay as Mul
    expr = sympy.parse_expr("a * b", evaluate=False)
    normalized = normalize_expression(expr)

    print(f"Simple multiplication: {expr} -> {normalized}")
    assert isinstance(normalized, sympy.Mul)


if __name__ == "__main__":
    test_division_normalization()
    test_subtraction_normalization()
    test_problematic_expression()
    test_mixed_operations()
    test_no_unwanted_changes()
    print("\\n✅ All core simplification tests passed!")
