import sympy


def test_sympy_simplify_example():
    """Verify that ``sympy.simplify`` rewrites a composite expression into its
    expected simplified rational form.

    The expression combines addition, subtraction, multiplication, and
    division.  We construct it exactly as in the user's example and check
    that ``sympy.simplify`` produces the canonical form with an expanded
    numerator.
    """

    x = sympy.Symbol("x")

    # Original composite expression
    expr = ((x + 2) * (x - 2) + (2 * x)) / (x - 2)

    # Expected simplified form – denominator remains the same but the
    # numerator is expanded and like terms are combined.
    expected = sympy.sympify("(x**2 + 2*x - 4)/(x - 2)")

    simplified = sympy.simplify(expr)

    # Sanity check that the expression did in fact change representation
    # (string comparison is sufficient for this deterministic construction)
    assert str(simplified) != str(
        expr
    ), "sympy.simplify did not alter the expression representation"

    # Mathematical equivalence check against the expected rational form.
    # We use SymPy's ``equals`` to avoid relying on exact string formatting.
    assert simplified.equals(expected), (
        "Simplified expression does not match the expected canonical form:"
        f"\nObtained:   {simplified}\nExpected:   {expected}"
    )
