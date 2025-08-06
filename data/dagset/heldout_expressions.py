# Heldout expressions and their corresponding sympy expressions for testing
import sympy


def parse_all(expressions):
    return [(expr, sympy.parse_expr(expr, evaluate=False)) for expr in expressions]


single_expressions = [
    "1.0",
    "1",
    "-1",
    "2.0",
    "2",
    "3.0",
    "3",
    "4.0",
    "4",
    "5.0",
    "5",
    "6.0",
    "-1.0",
    "0.5",
    "-0.5",
    "0.0",
    "0",
]

simple_expressions = [
    "1.0 + 1.0",
    "1 + 1",
    "1 + 1.0" "2.0 * 3.0",
    "10.0 - 5.0",
    "15.0 / 3.0",
    "-2.0 + 5.0",
    "-2 + 5",
    "3.0 * -4.0",
    "8.0 / -2.0",
]

complex_expressions = [
    "69.919 - 7.211",
    "-907.0 + 7.211",
    "2.0 * 3.0 + 4.0",
    "10.0 - 5.0 / 2.0",
    "10.0 - 5.0 / 2",
    "1.5 * 2.0 + 3.0",
    "8.0 / 2.0 - 1.0",
    "-3.0 + 4.0 * 2.0",
    "-3 + 4.3213 * 2",
    "12.0 / 3.0 + 5.0",
    "3.0 + 4.0 * 5.0 - 2.0",
    "10.0 / 2.0 + 3.0 * 4.0",
    "5.0 * 6.0 - 8.0 / 2.0",
    "15.0 - 3.0 * 2.0 + 7.0",
    "2.5 * 3.0 + 4.5 - 1.2",
    "12.0 / 4.0 + 2.5 * 3.0",
]

parentheses_expressions = [
    "(3.0 + 4.0) * 2.0",
    "(10.0 - 6.0) / 2.0",
    "2.0 * (3.0 + 4.0)",
    "(8.0 + 2.0) * (3.0 - 1.0)",
    "(15.0 - 3.0) / (2.0 + 1.0)",
    # Nested parentheses
    "((3.0 + 4.0) * 2.0) + 1.0",
    "(2.0 * (3.0 + 4.0)) - 5.0",
    "((10.0 - 6.0) / 2.0) * 3.0",
]

very_complex_expressions = [
    "-1487.475/(-2.0*69.919 + 5.0)",
    "-2.0 + (69.919 + (-907.0 + 7.211))",
    "3.5 + 4.2 * (1.8 - 6.7)",
    "(3.5 + 1.2) * 4.0 - 2.8",
    "15.5 / (2.5 + 1.0) + 3.2",
    "-5.3 + 8.0 * (2.1 - 4.7)",
    "(12.5 - 3.2) / (1.8 + 0.7)",
]

small_numbers_expressions = [
    "0.1 + 0.2",
    "0.01 * 0.5",
    "0.001 / 0.1",
    "1.0000001 - 1.0",
    "99.99999 + 0.00001",
]

large_numbers_expressions = [
    "100.0 + 200.0",
    "1000.0 * 2.0",
    "9999.0 / 3.0",
    "12345.6789 - 5432.1",
]

division_edge_cases_expressions = [
    "1.0 / 3.0",
    "2.0 / 7.0",
    "5.0 / 11.0",
    "22.0 / 7.0",
    "355.0 / 113.0",
]

integer_results_expressions = [
    "2.0 + 3.0",
    "6.0 - 2.0",
    "4.0 * 5.0",
    "20.0 / 4.0",
]

zero_handling_expressions = [
    "0.0 + 5.0",
    "5.0 - 5.0",
    "0.0 * 10.0",
    "0.0 / 5.0",
]

negative_number_combinations_expressions = [
    "-3.0 + 4.0",
    "5.0 + -2.0",
    "-10.0 * -3.0",
    "-15.0 / 3.0",
    "3.0 - -4.0",
    "-5.5 + 2.2",
]

mixed_integer_decimal_expressions = [
    "3 + 4.5",
    "7.2 - 2",
    "3 * 2.5",
    "10 / 2.5",
]

deeply_nested_expressions = [
    "((((1.0 + 2.0) * 3.0) - 4.0) / 5.0) + 6.0",
    "(1.0 + (2.0 + (3.0 + (4.0 + 5.0))))",
    "((1.0 * 2.0) * (3.0 * 4.0)) * 5.0",
]

realistic_complex_expressions = [
    "3.14159 * 2.0 + 1.41421",
    "(2.71828 + 1.61803) * 3.0",
    "sqrt(16.0) + sqrt(9.0)",
    "(1.0 / 3.0) * 9.0",
    "(2.0 / 3.0) * 6.0",
]

heldout_segments = {
    "single": parse_all(single_expressions),
    "simple": parse_all(simple_expressions),
    "complex": parse_all(complex_expressions),
    "parentheses": parse_all(parentheses_expressions),
    "very_complex": parse_all(very_complex_expressions),
    "small_numbers": parse_all(small_numbers_expressions),
    "large_numbers": parse_all(large_numbers_expressions),
    "division_edge_cases": parse_all(division_edge_cases_expressions),
    "integer_results": parse_all(integer_results_expressions),
    "zero_handling": parse_all(zero_handling_expressions),
    "negative_number_combinations": parse_all(negative_number_combinations_expressions),
    "mixed_integer_decimal": parse_all(mixed_integer_decimal_expressions),
    "deeply_nested": parse_all(deeply_nested_expressions),
    "realistic_complex": parse_all(realistic_complex_expressions),
}

heldout_expressions = [
    *single_expressions,
    *simple_expressions,
    *complex_expressions,
    *parentheses_expressions,
    *very_complex_expressions,
    *small_numbers_expressions,
    *large_numbers_expressions,
    *division_edge_cases_expressions,
    *integer_results_expressions,
    *zero_handling_expressions,
    *negative_number_combinations_expressions,
    *mixed_integer_decimal_expressions,
    *deeply_nested_expressions,
    *realistic_complex_expressions,
]

heldout_expressions = parse_all(heldout_expressions)
