#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterator, Tuple

import sympy
import torch
from num2words import num2words
from sympy import im, postorder_traversal
from sympy.core.numbers import Float as sympy_float
from tiktoken import get_encoding

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


# Custom SymPy operations for cleaner parsing
class Div(sympy.Function):
    """Custom division operation that can handle arbitrary arguments: Div(a, b, c) = a / b / c"""

    @classmethod
    def eval(cls, *args):
        """Don't evaluate - keep as symbolic custom operation"""
        if len(args) < 2:
            raise ValueError("Div requires at least 2 arguments")
        return None  # Return None to prevent evaluation

    def evalf(self, *args, **kwargs):
        """Evaluate numerically when needed"""
        result = self.args[0].evalf(*args, **kwargs)
        for arg in self.args[1:]:
            result = result / arg.evalf(*args, **kwargs)
        return result


class Sub(sympy.Function):
    """Custom subtraction operation that can handle arbitrary arguments: Sub(a, b, c) = a - b - c"""

    @classmethod
    def eval(cls, *args):
        """Don't evaluate - keep as symbolic custom operation"""
        if len(args) < 2:
            raise ValueError("Sub requires at least 2 arguments")
        return None  # Return None to prevent evaluation

    def evalf(self, *args, **kwargs):
        """Evaluate numerically when needed"""
        result = self.args[0].evalf(*args, **kwargs)
        for arg in self.args[1:]:
            result = result - arg.evalf(*args, **kwargs)
        return result


def isNum(node: sympy.Basic):
    return isinstance(node, sympy.Integer) or isinstance(node, sympy.Float)


def isUnnormalizedDivision(node: sympy.Basic):
    return (
        isinstance(node, sympy.Mul)
        and len(node.args) == 2
        and isinstance(node.args[1], sympy.Pow)
        and node.args[1].exp == -1
    )


# a + (-1 * b) -> a - b
def isUnnormalizedSubtraction(node: sympy.Basic):
    return (
        isinstance(node, sympy.Add)
        and len(node.args) == 2
        and isinstance(node.args[1], sympy.Mul)
        and node.args[1].args[0] == -1
    )


# -1 * x -> -x
def isUnnormalizedNegation(node: sympy.Basic):
    return isinstance(node, sympy.Mul) and len(node.args) == 2 and node.args[0] == -1


def normalize_expression(expr: sympy.Basic) -> sympy.Basic:
    """
    Normalize a SymPy expression by converting complex patterns into custom operations.
    """

    def transform_node(node):
        # a * b^(-1) -> a / b
        if isUnnormalizedDivision(node):
            return Div(
                transform_node(node.args[0]),
                transform_node(node.args[1].base),
                evaluate=False,
            )

        # a + (-1 * b) -> a - b
        if isUnnormalizedSubtraction(node):
            return Sub(
                transform_node(node.args[0]),
                transform_node(node.args[1].args[1]),
                evaluate=False,
            )

        # -1 * x -> -x
        if isUnnormalizedNegation(node):
            return -transform_node(node.args[1])

        # Leaf nodes are returned as-is
        if isNum(node) or isinstance(node, sympy.Symbol):
            return node

        # Only Mul and Add operations should exist.
        if not isinstance(node, sympy.Mul) and not isinstance(node, sympy.Add):
            raise ValueError(f"Unexpected node: {node}")

        # Transform all arguments
        _type = type(node)
        return _type(*[transform_node(arg) for arg in node.args], evaluate=False)

    return transform_node(expr)


def float_to_digit_onehot(
    value: float, max_digits: int, max_decimal_places: int, base: int = 10
) -> torch.Tensor:
    """Convert a float into a one-hot tensor of shape (D, base) where D = max_digits + max_decimal_places.

    The function validates that the number's magnitude can be represented within the specified digits
    and decimal places. The integer part is left-padded with zeros and the fractional part is
    right-padded with zeros so that their total length equals max_digits + max_decimal_places.

    Args:
        value: The float value to convert
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places
        base: Number base for digit representation (2-36 supported, defaults to 10)

    Returns:
        One-hot tensor of shape (max_digits + max_decimal_places, base)

    Raises:
        ValueError: If inputs are invalid, value magnitude exceeds representable range, or conversion fails
    """
    # Validate inputs
    if not isinstance(value, (int, float)):
        raise ValueError(f"Value must be a number, got {type(value)}")

    if not torch.isfinite(torch.tensor(float(value))):
        raise ValueError(f"Value must be finite (not NaN or inf), got {value}")

    if not isinstance(max_digits, int) or max_digits <= 0:
        raise ValueError(f"max_digits must be a positive integer, got {max_digits}")

    if not isinstance(max_decimal_places, int) or max_decimal_places <= 0:
        raise ValueError(
            f"max_decimal_places must be a positive integer, got {max_decimal_places}"
        )

    if not isinstance(base, int) or base < 2 or base > 36:
        raise ValueError(f"Base must be an integer between 2 and 36, got {base}")

    # Check if magnitude is too large for the available digits
    limit = base**max_digits - base ** (-max_decimal_places)
    abs_val = abs(value)
    if abs_val > limit:
        raise ValueError(
            f"Value magnitude {abs_val} exceeds maximum representable value {limit} "
            f"for {max_digits} digits and {max_decimal_places} decimal places in base {base}"
        )

    if base == 10:
        # Use string formatting to avoid floating point precision issues for base 10
        format_str = f"{{:.{max_decimal_places}f}}"
        value_str = format_str.format(abs_val)

        # Split into integer and decimal parts
        if "." in value_str:
            int_part_str, frac_part_str = value_str.split(".")
        else:
            int_part_str = value_str
            frac_part_str = ""

        # Pad integer part to max_digits (left pad with zeros)
        int_part_str = int_part_str.zfill(max_digits)[-max_digits:]

        # Pad fractional part to max_decimal_places (right pad with zeros)
        frac_part_str = (frac_part_str + "0" * max_decimal_places)[:max_decimal_places]

        # Combine and convert to digit list
        all_digits_str = int_part_str + frac_part_str
        all_digits = [int(d) for d in all_digits_str]
    else:
        # For non-base-10, use the original method but with rounding to mitigate precision issues
        # Convert to the target base
        # First convert integer part
        int_part = int(abs_val)
        frac_part = abs_val - int_part

        # Convert integer part to target base
        if int_part == 0:
            int_digits = [0]
        else:
            int_digits = []
            temp = int_part
            while temp > 0:
                int_digits.append(temp % base)
                temp = temp // base
            int_digits.reverse()  # Most significant digit first

        # Pad or truncate integer part to exactly max_digits
        if len(int_digits) > max_digits:
            int_digits = int_digits[-max_digits:]  # Keep least significant digits
        else:
            int_digits = [0] * (
                max_digits - len(int_digits)
            ) + int_digits  # Pad with zeros

        # Convert fractional part to target base with rounding
        frac_digits = []
        temp_frac = frac_part
        for _ in range(max_decimal_places):
            temp_frac *= base
            # Add small epsilon and round to mitigate floating point precision issues
            digit = (
                round(temp_frac + 1e-10) if temp_frac < base - 0.5 else int(temp_frac)
            )
            digit = min(max(digit, 0), base - 1)  # Clamp to valid range
            frac_digits.append(digit)
            temp_frac -= digit

        # Combine integer and fractional digits
        all_digits = int_digits + frac_digits

    D = max_digits + max_decimal_places
    if len(all_digits) != D:
        raise ValueError(
            f"Digit conversion failed: expected {D} digits, got {len(all_digits)}"
        )

    # Validate all digits are in valid range
    for i, digit in enumerate(all_digits):
        if not (0 <= digit < base):
            raise ValueError(
                f"Invalid digit {digit} at position {i}, must be in range [0, {base-1}]"
            )

    # Create one-hot encoding
    one_hot = torch.zeros(D, base)
    for i, digit in enumerate(all_digits):
        one_hot[i, digit] = 1.0

    # Validate output tensor
    if not torch.allclose(one_hot.sum(dim=1), torch.ones(D)):
        raise ValueError(
            "Failed to create valid one-hot encoding: each row must sum to 1"
        )

    if one_hot.shape != (D, base):
        raise ValueError(
            f"Output tensor has wrong shape: expected ({D}, {base}), got {one_hot.shape}"
        )

    return one_hot


def digit_onehot_to_float(
    digit_onehot: torch.Tensor, sign: float, max_digits: int, max_decimal_places: int
) -> float:
    """Convert digit one-hot tensor back to float value.

    Args:
        digit_onehot: One-hot tensor of shape (D, base)
        sign: Sign of the number (+1 or -1)
        max_digits: Number of integer digit positions
        max_decimal_places: Number of decimal digit positions

    Returns:
        Float value reconstructed from digits

    Raises:
        ValueError: If inputs are invalid or conversion fails
    """
    # Validate inputs
    if not isinstance(digit_onehot, torch.Tensor):
        raise ValueError(
            f"digit_onehot must be a torch.Tensor, got {type(digit_onehot)}"
        )

    if digit_onehot.dim() != 2:
        raise ValueError(
            f"digit_onehot must be 2D tensor, got {digit_onehot.dim()}D with shape {digit_onehot.shape}"
        )

    D = max_digits + max_decimal_places
    if digit_onehot.shape[0] != D:
        raise ValueError(
            f"digit_onehot must have {D} rows (max_digits + max_decimal_places), got {digit_onehot.shape[0]}"
        )

    base = digit_onehot.shape[1]
    if base < 2 or base > 36:
        raise ValueError(f"Invalid base {base}, must be between 2 and 36")

    if not isinstance(sign, (int, float)):
        raise ValueError(f"sign must be a number, got {type(sign)}")

    if sign not in [-1, 0, 1]:
        raise ValueError(f"sign must be -1, 0, or 1, got {sign}")

    if not isinstance(max_digits, int) or max_digits <= 0:
        raise ValueError(f"max_digits must be a positive integer, got {max_digits}")

    if not isinstance(max_decimal_places, int) or max_decimal_places <= 0:
        raise ValueError(
            f"max_decimal_places must be a positive integer, got {max_decimal_places}"
        )

    # Validate that each row is a valid one-hot vector
    row_sums = digit_onehot.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        invalid_rows = torch.nonzero(
            ~torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        )
        raise ValueError(
            f"Invalid one-hot encoding: rows {invalid_rows.flatten().tolist()} do not sum to 1"
        )

    # Check that values are non-negative (one-hot should have 0s and 1s)
    if (digit_onehot < 0).any() or (digit_onehot > 1).any():
        raise ValueError(
            "digit_onehot values must be between 0 and 1 (one-hot encoding)"
        )

    # Get the digit indices from one-hot encoding
    digit_indices = torch.argmax(digit_onehot, dim=1)

    # Validate digit indices are in valid range
    if (digit_indices < 0).any() or (digit_indices >= base).any():
        raise ValueError(f"Invalid digit indices: must be in range [0, {base-1}]")

    try:
        # Convert to string representation
        digit_str = "".join(str(d.item()) for d in digit_indices)

        # Split into integer and decimal parts
        int_part = digit_str[:max_digits]
        dec_part = digit_str[max_digits:]

        # Remove leading zeros from integer part, but keep at least one digit
        int_part_clean = str(int(int_part))

        # Combine parts and convert to float
        value_str = f"{int_part_clean}.{dec_part}"
        value = float(value_str)

        # Validate the resulting value is finite
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Conversion resulted in non-finite value: {value}")

        # Apply sign
        final_value = value * sign

        # Final validation
        if not torch.isfinite(torch.tensor(final_value)):
            raise ValueError(f"Final value is not finite: {final_value}")

        return final_value

    except (ValueError, OverflowError) as e:
        if isinstance(e, ValueError) and "non-finite" in str(e):
            raise e  # Re-raise our own validation errors
        raise ValueError(f"Failed to convert digits to float: {e}") from e


def expression_to_tensors(
    expr: sympy.Basic, dag_depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a SymPy expression to DAG tensor representation.

    Returns:
        V_mag: (1, 1, total_nodes) - magnitudes of all nodes (initial + intermediate)
        V_sign: (1, 1, total_nodes) - signs of all nodes
        O: (1, 1, dag_depth, total_nodes) - operand selection matrix
        G: (1, 1, dag_depth) - domain selector (0=log, 1=linear)
    """
    # Step 1: Normalize the expression
    normalized_expr = normalize_expression(expr)

    # import pdb; pdb.set_trace()
    num_initial_nodes = dag_depth + 1
    num_intermediate_nodes = dag_depth
    total_nodes = num_initial_nodes + num_intermediate_nodes

    V_mag = torch.zeros(1, 1, total_nodes)
    V_sign = torch.ones(1, 1, total_nodes)  # Initialize to all 1s
    O = torch.zeros(1, 1, dag_depth, total_nodes)
    G = torch.ones(1, 1, dag_depth)  # Initialize to all 1s (linear domain)

    # Track values list and step index
    values = []
    step_index = 0
    node_replacements = {}  # For Neg of numbers

    # PHASE 1: Collect all initial numeric values (including Neg of numbers)
    for node in postorder_traversal(normalized_expr):
        is_float = isinstance(node, sympy_float)
        is_int = isinstance(node, sympy.Integer)
        is_rational = isinstance(node, sympy.Rational)

        # Handle direct numbers
        if is_float or is_int or is_rational:
            if step_index >= num_initial_nodes:
                raise ValueError(
                    f"Expression has too many initial values. Needs dag_depth >= {step_index - 1} (allowing {num_initial_nodes} initial values)"
                )

            try:
                val = float(node.evalf())
                V_mag[0, 0, step_index] = abs(val)
                V_sign[0, 0, step_index] = 1 if val >= 0 else -1
                values.append(node)
                step_index += 1
            except (TypeError, ValueError):
                raise ValueError(f"Cannot convert {node} to float")

        # Error checks
        elif isUnnormalizedNegation(node):
            raise ValueError(
                f"Unexpected old-style negation after normalization: {node}. "
                f"All Mul(-1, x) patterns should have been converted to -x during normalization. "
                f"Normalization failed to catch this pattern."
            )
        elif isUnnormalizedDivision(node):
            raise ValueError(
                f"Unexpected old-style division after normalization: {node}. "
                f"All Mul(a, Pow(b, -1)) patterns should have been converted to Div(a, b) during normalization. "
                f"Normalization failed to catch this pattern."
            )
        elif isUnnormalizedSubtraction(node):
            raise ValueError(
                f"Unexpected old-style subtraction after normalization: {node}. "
                f"All Add(a, Mul(-1, b)) patterns should have been converted to Sub(a, b) during normalization. "
                f"Normalization failed to catch this pattern."
            )
        elif isinstance(node, sympy.Pow):
            raise ValueError(
                f"Unexpected Pow node after normalization: {node}. "
                f"All Pow(x, -1) should be converted to Div(1, x). "
                f"Other Pow operations are not supported."
            )

    # Pad values list to num_initial_nodes
    while len(values) < num_initial_nodes:
        values.append("UNUSED")

    # Reset step_index for operations
    step_index = 0

    # PHASE 2: Process operations in second postorder traversal
    for node in postorder_traversal(normalized_expr):
        # Skip leaf nodes (already processed in Phase 1)
        is_float = isinstance(node, sympy_float)
        is_int = isinstance(node, sympy.Integer)
        is_rational = isinstance(node, sympy.Rational)
        if is_float or is_int or is_rational:
            continue

        if step_index >= dag_depth:
            raise ValueError(
                f"Expression requires more operations than dag_depth={dag_depth}, num operations: {step_index}"
            )

        is_div = isinstance(node, Div)
        is_sub = isinstance(node, Sub)
        is_mul = isinstance(node, sympy.Mul)
        is_add = isinstance(node, sympy.Add)

        # Set domain: log for Div/Mul, linear for Add/Sub/Neg
        if is_div or is_mul:
            G[0, 0, step_index] = 0  # Log domain
        elif is_add or is_sub:
            G[0, 0, step_index] = 1  # Linear domain
        elif is_div or is_sub:
            first_coefficient = 1
            coefficient = -1
        elif is_add or is_mul:
            first_coefficient = 1
            coefficient = 1

        # Process arguments and set operand matrix
        if is_div or is_sub or is_add or is_mul:
            for i, arg in enumerate(node.args):
                arg = node_replacements.get(arg, arg)  # Use replacement if exists
                arg_idx = values.index(arg)  # Find index of this argument
                _coefficient = first_coefficient if i == 0 else coefficient
                O[0, 0, step_index, arg_idx] += _coefficient  # Accumulate coefficients

            values.append(node)
            step_index += 1
        else:
            raise ValueError(f"Unexpected node type: {type(node)}")

            # Fill remaining steps with identity operations
    if step_index > 0:
        # Had operations - use the last operation result slot
        # The executor stores operation results at num_initial_nodes + step_number
        last_result_slot = num_initial_nodes + (step_index - 1)
        for remaining_step in range(step_index, dag_depth):
            O[0, 0, remaining_step, last_result_slot] = 1
    else:
        # No operations - use the first initial value
        for remaining_step in range(0, dag_depth):
            O[0, 0, remaining_step, 0] = 1

    return V_mag, V_sign, O, G


def tensor_to_expression(
    digit_logits: torch.Tensor,  # (num_initial_nodes, D, base)
    V_sign: torch.Tensor,  # (total_nodes,)
    O: torch.Tensor,  # (dag_depth, total_nodes)
    G: torch.Tensor,  # (dag_depth,)
    max_digits: int = 4,
    max_decimal_places: int = 4,
) -> sympy.Basic:
    """
    Convert DAG tensors back to a SymPy expression.

    Args:
        digit_logits: Digit predictions for initial nodes
        V_sign: Signs of all nodes
        O: Operand selection matrix
        G: Domain gates (0=log space, 1=linear space)
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places

    Returns:
        SymPy expression representing the DAG computation
    """
    import sympy

    from .streaming import digit_onehot_to_float

    # Get tensor dimensions
    num_initial_nodes = digit_logits.shape[0]
    dag_depth = O.shape[0]
    total_nodes = O.shape[1]

    # Convert digit logits to actual values for initial nodes
    initial_values = []
    for n in range(num_initial_nodes):
        # Convert logits to probabilities and then to one-hot
        digit_probs = torch.softmax(digit_logits[n], dim=-1)
        digit_indices = torch.argmax(digit_probs, dim=1)

        # Create one-hot encoding
        digit_onehot = torch.zeros_like(digit_probs)
        for d, idx in enumerate(digit_indices):
            digit_onehot[d, idx] = 1.0

        # Get sign for this node and convert to discrete sign
        sign_continuous = V_sign[n].item()
        sign = 1.0 if sign_continuous >= 0 else -1.0

        # Convert to float value
        float_value = digit_onehot_to_float(
            digit_onehot, sign, max_digits, max_decimal_places
        )

        # Create SymPy number
        if abs(float_value - round(float_value)) < 1e-10:
            # Integer value
            initial_values.append(sympy.Integer(int(round(float_value))))
        else:
            # Float value
            initial_values.append(sympy.Float(float_value))

    # Track intermediate results
    node_expressions = initial_values.copy()

    # Process each operation step
    for step in range(dag_depth):
        # Get operand coefficients for this step
        operand_coeffs = O[step]  # (total_nodes,)
        domain_gate = G[step].item()  # 0=log, 1=linear

        # Find non-zero operands
        nonzero_indices = torch.nonzero(operand_coeffs, as_tuple=False).flatten()

        if len(nonzero_indices) == 0:
            # No operation, just pass through the result from previous step
            if step > 0:
                # Use result from previous operation
                result_expr = node_expressions[num_initial_nodes + step - 1]
            else:
                # Use first initial value
                result_expr = node_expressions[0]
        elif len(nonzero_indices) == 1:
            # Unary operation (negation)
            idx = nonzero_indices[0].item()
            coeff = operand_coeffs[idx].item()

            if idx < len(node_expressions):
                operand = node_expressions[idx]
                if abs(coeff + 1) < 1e-10:  # coefficient ≈ -1
                    result_expr = -operand
                else:
                    result_expr = coeff * operand
            else:
                # Invalid index, use fallback
                result_expr = (
                    node_expressions[0] if node_expressions else sympy.Integer(0)
                )

        elif len(nonzero_indices) == 2:
            # Binary operation
            idx1, idx2 = nonzero_indices[0].item(), nonzero_indices[1].item()
            coeff1, coeff2 = operand_coeffs[idx1].item(), operand_coeffs[idx2].item()

            # Check if indices are valid
            valid_operands = []
            if idx1 < len(node_expressions):
                valid_operands.append((node_expressions[idx1], coeff1))
            if idx2 < len(node_expressions):
                valid_operands.append((node_expressions[idx2], coeff2))

            if len(valid_operands) == 2:
                operand1, coeff1 = valid_operands[0]
                operand2, coeff2 = valid_operands[1]

                # Determine operation type based on domain and coefficients
                if domain_gate == 0:  # Log domain (Mul/Div)
                    if abs(coeff1 - 1) < 1e-10 and abs(coeff2 - 1) < 1e-10:
                        result_expr = operand1 * operand2
                    elif abs(coeff1 - 1) < 1e-10 and abs(coeff2 + 1) < 1e-10:
                        result_expr = operand1 / operand2
                    else:
                        # General case with coefficients
                        if coeff2 > 0:
                            result_expr = operand1 * (operand2**coeff2)
                        else:
                            result_expr = operand1 / (operand2 ** abs(coeff2))
                else:  # Linear domain (Add/Sub)
                    if abs(coeff1 - 1) < 1e-10 and abs(coeff2 - 1) < 1e-10:
                        result_expr = operand1 + operand2
                    elif abs(coeff1 - 1) < 1e-10 and abs(coeff2 + 1) < 1e-10:
                        result_expr = operand1 - operand2
                    else:
                        # General case with coefficients
                        result_expr = coeff1 * operand1 + coeff2 * operand2
            elif len(valid_operands) == 1:
                # Only one valid operand - treat as unary
                operand, coeff = valid_operands[0]
                if abs(coeff + 1) < 1e-10:
                    result_expr = -operand
                else:
                    result_expr = coeff * operand
            else:
                # No valid operands - use fallback
                result_expr = (
                    node_expressions[0] if node_expressions else sympy.Integer(0)
                )
        else:
            # More than 2 operands - sum/product them
            terms = []
            for idx in nonzero_indices:
                idx_val = idx.item()
                coeff = operand_coeffs[idx_val].item()

                # Skip operands that refer to nodes we haven't computed yet
                if idx_val >= len(node_expressions):
                    continue

                operand = node_expressions[idx_val]

                if abs(coeff - 1) < 1e-10:
                    terms.append(operand)
                elif abs(coeff + 1) < 1e-10:
                    terms.append(-operand)
                else:
                    terms.append(coeff * operand)

            if len(terms) == 0:
                # No valid operands, use first initial value as fallback
                result_expr = (
                    node_expressions[0] if node_expressions else sympy.Integer(0)
                )
            elif len(terms) == 1:
                result_expr = terms[0]
            elif domain_gate == 0:  # Log domain - multiply
                result_expr = sympy.Mul(*terms)
            else:  # Linear domain - add
                result_expr = sympy.Add(*terms)

        # Store intermediate result
        if len(node_expressions) <= num_initial_nodes + step:
            node_expressions.append(result_expr)
        else:
            node_expressions[num_initial_nodes + step] = result_expr

    # Return the final expression (last intermediate result or first value if no operations)
    if dag_depth > 0 and len(node_expressions) > num_initial_nodes:
        return node_expressions[-1]
    return node_expressions[0] if node_expressions else sympy.Integer(0)


# Import expression generation functionality
from .generate_expression import (
    generate_expression,
)

# ============================================================================
# GENERATION SYSTEM: Separation between Generation and State Operations
# ============================================================================

# ============================================================================

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


@dataclass
class DAGExample:
    """Base class for DAG computation examples with essential shared attributes."""

    text: str
    structure_dict: dict[str, torch.Tensor]
    depth: int
    max_decimal_places: int
    seed: int

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}(seed={self.seed}, text={self.text}, depth={self.depth})"


@dataclass
class DAGTrainExample(DAGExample):
    """Lightweight DAG example for training with minimal attributes to reduce memory overhead."""


@dataclass
class DAGValExample(DAGExample):
    """Full DAG example for validation with all attributes for logging and debugging.

    Note: This represents N different expressions (per-token), where the fields below
    correspond to the final/complete expression in the sequence.
    """

    full_operations_named: list[str]  # Operations for the final complete expression
    full_operations: list[str]  # Operations for the final complete expression
    final_value_sympy: float | None = None
    full_expr: sympy.Basic | None = None  # The final complete expression

    def __str__(self):
        V_mag_shape = self.structure_dict["target_V_mag"].shape
        V_sign_shape = self.structure_dict["target_V_sign"].shape
        O_shape = self.structure_dict["target_O"].shape
        G_shape = self.structure_dict["target_G"].shape
        final_value_exec = self.structure_dict["target_final_exec"]
        return f"DAGValExample(seed={self.seed}, text={self.text}, depth={self.depth}, V_mag={V_mag_shape}, V_sign={V_sign_shape}, O={O_shape}, G={G_shape}, full_operations_named={self.full_operations_named}, final_value_sympy={self.final_value_sympy}, final_value_exec={final_value_exec}, full_expr={self.full_expr})"


def expression_to_string(expr: sympy.Basic) -> str:
    """Convert a sympy expression to string."""
    return str(expr)


def convert_number_to_english(number: float, max_decimal_places: int = 6) -> str:
    """Convert *number* to its English word equivalent using *num2words*.

    The value is first rounded (half-up) to *max_decimal_places* decimal digits to
    avoid extremely long fractional strings, then converted.  Negatives are
    rendered with the "negative" prefix to preserve the previous output style.
    """
    # Quantise using Decimal to avoid floating-point surprises (e.g. 0.1+0.2)
    quantised = Decimal(str(number)).quantize(
        Decimal(10) ** -max_decimal_places, rounding=ROUND_HALF_UP
    )

    words = num2words(abs(quantised))
    return f"negative {words}" if quantised < 0 else words


def expressions_to_tensors(
    expressions: list[sympy.Basic | str],
    *,
    depth: int,
    max_digits: int = 4,
    max_decimal_places: int = 4,
) -> tuple[list[dict[str, torch.Tensor]], list[bool]]:
    """Convert a list of sympy expressions to structure tensors for the digit prediction system.

    This converts SymPy expressions to DAG tensor format:
    - Uses digit prediction, V_sign, O, G tensor representation
    - Invalid expressions get "zero DAG" representations

    Args:
        expressions: List of sympy expressions or "not valid" strings
        depth: Target depth for DAG operations
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places

    Returns:
        Tuple of (tensor_list, valid_mask) where:
        - tensor_list: Contains T tensors (one per token position, including zero DAGs)
        - valid_mask: Boolean list indicating which positions were valid

    Tensor shapes:
        - target_digits: (num_initial_nodes, D, base)
        - target_V_sign: (total_nodes,)
        - target_O: (dag_depth, total_nodes)
        - target_G: (dag_depth,)
    """
    tensor_results = []
    valid_mask = []
    for expr in expressions:
        if expr == "not valid":
            # Create zero DAG for invalid token position with new architecture
            num_initial_nodes = depth + 1
            total_nodes = (depth + 1) + depth
            D = max_digits + max_decimal_places

            # Create zero digit targets (proper one-hot for digit 0)
            base = 10  # Default base
            zero_digits = torch.zeros(num_initial_nodes, D, base)
            # Set all digit positions to represent 0 (valid one-hot)
            zero_digits[:, :, 0] = 1.0

            zero_tensor_dict = {
                "target_digits": zero_digits,
                "target_V_sign": torch.ones(total_nodes),
                "target_O": torch.zeros(depth, total_nodes),
                "target_G": torch.ones(depth),
                "target_final_exec": 0.0,
            }
            tensor_results.append(zero_tensor_dict)
            valid_mask.append(False)
        else:
            # Convert SymPy expression directly to DAG tensor format
            try:
                V_mag, V_sign, O, G = expression_to_tensors(expr, depth)

                # Compute the actual target execution value by evaluating the SymPy expression
                try:
                    target_final_exec_value = float(expr.evalf())
                except:
                    print(f"Evaluation failed for expression: {expr}")
                    raise

                # Convert V_mag values to digit targets for initial nodes only
                num_initial_nodes = depth + 1
                D = max_digits + max_decimal_places
                base = 10  # Default base
                target_digits = torch.zeros(num_initial_nodes, D, base)

                # Extract V_mag values and convert to digits for initial nodes
                V_mag_squeezed = V_mag.squeeze(0).squeeze(0)  # (total_nodes,)
                V_sign_squeezed = V_sign.squeeze(0).squeeze(0)  # (total_nodes,)

                for n in range(num_initial_nodes):
                    mag_value = V_mag_squeezed[n].item()
                    sign_value = V_sign_squeezed[n].item()

                    # Reconstruct the actual float value (mag * sign)
                    actual_value = mag_value * sign_value

                    # Convert to digit one-hot representation
                    digit_onehot = float_to_digit_onehot(
                        actual_value, max_digits, max_decimal_places, base
                    )
                    target_digits[n] = digit_onehot

                # Convert to target format for training
                # Remove batch and time dimensions (convert from (1,1,X) to (X))
                target_dict = {
                    "target_digits": target_digits,  # (num_initial_nodes, D, base)
                    "target_V_sign": V_sign_squeezed,  # (total_nodes,)
                    "target_O": O.squeeze(0).squeeze(0),  # (dag_depth, total_nodes)
                    "target_G": G.squeeze(0).squeeze(0),  # (dag_depth,)
                    "target_final_exec": target_final_exec_value,  # Actual execution result
                }
                tensor_results.append(target_dict)
                valid_mask.append(True)
            except:
                print(f"Conversion failed for expression: {expr}")
                raise

    return tensor_results, valid_mask


# ============================================================================
# DATA LOADING
# ============================================================================


def create_dag_structure_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_depth: int = 8,
    seed: int = 42,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    block_size: int = 8,
) -> Tuple[Iterator, Iterator]:
    """Create per-token train/val DAG structure dataloaders.

    Uses the streaming.py per-token system directly.

    Returns:
        Tuple of (train_loader, val_loader) that yield:
        - texts: List[str]
        - target_tensors: List[Dict[str, torch.Tensor]]
        - valid_masks: torch.Tensor (B, T)
    """

    def generate_batch(batch_size: int, training_seed: int):
        """Generate a batch using the per-token format."""
        tokenizer = get_encoding("gpt2")

        all_texts = []
        all_target_tensors = []
        all_valid_masks = []

        for i in range(batch_size):
            # Generate expressions
            expressions, substrings, _ = generate_expression(
                depth=max_depth,
                seed=training_seed + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=tokenizer,
            )

            # Convert to tensors
            target_tensors, tensor_valid_mask = expressions_to_tensors(
                expressions,
                depth=max_depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            # Get main expression (last one) as string
            last_expr = expressions[-1] if expressions else ""
            if last_expr == "not valid":
                text = (
                    substrings[-1] if substrings else ""
                )  # Use raw substring for invalid expressions
            else:
                text = str(last_expr)  # Convert sympy expression to string

            all_texts.append(text)
            all_target_tensors.append(
                target_tensors
            )  # Keep as list of lists (batch structure)
            all_valid_masks.append(tensor_valid_mask)

        # Pad valid masks to block_size (not just max within batch)
        batched_valid_masks = []
        for mask in all_valid_masks:
            # Truncate or pad to block_size to match model expectations
            if len(mask) > block_size:
                padded = mask[:block_size]  # Truncate if too long
            else:
                padded = mask + [False] * (block_size - len(mask))  # Pad with False
            batched_valid_masks.append(padded)

        valid_masks_tensor = torch.tensor(batched_valid_masks, dtype=torch.bool)

        # Pad target tensors to block_size as well
        padded_target_tensors = []
        for target_tensors in all_target_tensors:
            # Create zero DAG for padding positions using new tensor format
            total_nodes = (max_depth + 1) + max_depth
            zero_tensor_dict = {
                "target_V_mag": torch.zeros(total_nodes),
                "target_V_sign": torch.ones(total_nodes),
                "target_O": torch.zeros(max_depth, total_nodes),
                "target_G": torch.ones(max_depth),
                "target_final_exec": 0.0,
            }

            # Truncate or pad to block_size
            if len(target_tensors) > block_size:
                padded_tensors = target_tensors[:block_size]  # Truncate if too long
            else:
                padded_tensors = target_tensors + [zero_tensor_dict] * (
                    block_size - len(target_tensors)
                )
            padded_target_tensors.append(padded_tensors)

        return all_texts, padded_target_tensors, valid_masks_tensor

    def train_loader():
        counter = 0
        while True:
            texts, target_tensors, valid_masks = generate_batch(
                train_batch_size, seed + counter
            )
            counter += train_batch_size
            yield texts, target_tensors, valid_masks

    def val_loader():
        counter = 0
        while True:
            texts, target_tensors, valid_masks = generate_batch(
                val_batch_size, seed + counter + 10000
            )
            counter += val_batch_size
            yield texts, target_tensors, valid_masks

    return train_loader(), val_loader()
