#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import sympy
import torch
from sympy import postorder_traversal
from tiktoken import get_encoding

from .generate_expression import (
    generate_expressions,
)

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


# Custom SymPy operations we use to track the DAG structure
# Since for these operations we only need to change the
# selector coefficients and do not need to add a +-1 initial
# value and extra operation.
class Neg(sympy.Function):
    """Custom negation operation: Neg(x) = -x"""

    @classmethod
    def eval(cls, arg):
        # Don't evaluate automatically - keep as Neg object
        return None


class Recip(sympy.Function):
    """Custom reciprocal operation: Recip(x) = 1 / x"""

    @classmethod
    def eval(cls, arg):
        # Don't evaluate automatically - keep as Recip object
        return None


def isNum(node: sympy.Basic):
    return (
        isinstance(node, sympy.Integer)
        or isinstance(node, sympy.Float)
        or isinstance(node, sympy.Rational)
        or isinstance(node, sympy.Number)
    )


def isUnnormalizedNegation(node: sympy.Basic):
    return isinstance(node, sympy.Mul) and len(node.args) == 2 and node.args[0] == -1


def isUnnormalizedDivision(node: sympy.Basic):
    return isinstance(node, sympy.Pow) and node.exp == -1


def normalize_expression(expr: sympy.Basic) -> sympy.Basic:
    """
    Normalize a SymPy expression by converting complex patterns into custom operations.
    """

    def transform_node(node):
        # -1 * x -> Neg(x)
        if isUnnormalizedNegation(node):
            return Neg(transform_node(node.args[1]))

        # b^-1 -> Recip(b)
        if isUnnormalizedDivision(node):
            return Recip(transform_node(node.base))

        # Leaf nodes are returned as-is
        if isNum(node) or isinstance(node, sympy.Symbol):
            return node

        # Only Mul and Add operations should exist.
        if (
            not isinstance(node, sympy.Mul)
            and not isinstance(node, sympy.Add)
            and not isinstance(node, Neg)
            and not isinstance(node, Recip)
        ):
            raise ValueError(f"Unexpected node: {node}, type: {type(node)}")

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

    # Convert to the target base using mathematical approach (works for all bases)
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
        int_digits = [0] * (max_digits - len(int_digits)) + int_digits  # Pad with zeros

    # Convert fractional part to target base with rounding
    frac_digits = []
    temp_frac = frac_part
    for _ in range(max_decimal_places):
        temp_frac *= base
        # Extract the integer part as the digit
        digit = int(temp_frac)
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

    except Exception as e:
        raise ValueError(f"Failed to convert digits to float: {e}") from e


def expression_to_tensors(
    expr: sympy.Basic, dag_depth: int, max_digits: int = 4, max_decimal_places: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a SymPy expression to DAG tensor representation.

    Returns:
        target_digits: (1, 1, num_initial_nodes, D, base) - digit one-hot encodings for initial nodes
        V_sign: (1, 1, total_nodes) - signs of all nodes
        O: (1, 1, dag_depth, total_nodes) - operand selection matrix
        G: (1, 1, dag_depth) - domain selector (0=log, 1=linear)
    """
    # Step 1: Normalize the expression
    normalized_expr = normalize_expression(expr)

    num_initial_nodes = dag_depth + 1
    num_intermediate_nodes = dag_depth
    total_nodes = num_initial_nodes + num_intermediate_nodes
    D = max_digits + max_decimal_places
    base = 10  # Default base

    # Initialize target_digits tensor for initial nodes only
    target_digits = torch.zeros(1, 1, num_initial_nodes, D, base)
    V_sign = torch.ones(1, 1, total_nodes)  # Initialize to all 1s
    O = torch.zeros(1, 1, dag_depth, total_nodes)
    G = torch.ones(1, 1, dag_depth)  # Initialize to all 1s (linear domain)

    # Track values list and step index
    values = []
    step_index = 0

    # PHASE 1: Collect all initial numeric values
    for node in postorder_traversal(normalized_expr):
        # Handle direct numbers
        if isNum(node) and node not in values:
            if step_index >= num_initial_nodes:
                raise ValueError(
                    f"Expression has too many initial values. Needs dag_depth >= {step_index - 1} (allowing {num_initial_nodes} initial values)"
                )

            try:
                val = float(node.evalf())
                # Convert to digit one-hot encoding
                digit_onehot = float_to_digit_onehot(
                    abs(val), max_digits, max_decimal_places, base
                )
                target_digits[0, 0, step_index] = digit_onehot
                V_sign[0, 0, step_index] = 1 if val >= 0 else -1
                values.append(node)
                step_index += 1
            except (TypeError, ValueError):
                raise ValueError(f"Cannot convert {node} to float")
        # Error checks
        elif isUnnormalizedNegation(node):
            raise ValueError(
                f"Unexpected old-style negation after normalization: {node}. "
                f"All Mul(-1, x) patterns should have been converted to Neg(x) during normalization. "
                f"Normalization failed to catch this pattern."
            )
        elif isUnnormalizedDivision(node):
            raise ValueError(
                f"Unexpected old-style division after normalization: {node}. "
                f"All Mul(a, Pow(b, -1)) patterns should have been converted to Div(a, b) during normalization. "
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

    # Neg and Recip are encoded in the coefficient, so they should be considered no-ops.
    # However, when they appear as operands of other operations, we need to map them
    # to their actual values that exist in the values list.
    operand_map = {}  # Maps operands to their actual values for lookup

    # PHASE 2: Process operations in second postorder traversal
    for node in postorder_traversal(normalized_expr):
        is_neg = isinstance(node, Neg)
        is_recip = isinstance(node, Recip)

        # Skip leaf nodes (already processed in Phase 1)
        if isNum(node) or is_neg or is_recip:
            continue

        if step_index >= dag_depth:
            raise ValueError(
                f"Expression requires more operations than dag_depth={dag_depth}, num operations: {step_index}"
            )

        is_mul = isinstance(node, sympy.Mul)
        is_add = isinstance(node, sympy.Add)

        # Set domain
        if is_mul:
            G[0, 0, step_index] = 0  # Log domain
        elif is_add:
            G[0, 0, step_index] = 1  # Linear domain

        # Process arguments and extract operands with coefficients
        operands = []
        coefficients = []

        for arg in node.args:
            # This prevents us from getting "not in list" errors by mapping Neg/Recip
            # operands to their inner values that actually exist in the values list
            if isinstance(arg, Neg) or isinstance(arg, Recip):
                coefficients.append(-1)
                # Recursively resolve nested Neg/Recip to find the actual value
                inner = arg.args[0]
                while isinstance(inner, (Neg, Recip)):
                    inner = inner.args[0]
                operand_map[arg] = (
                    inner  # Map to ultimate inner operand (e.g., Neg(Neg(5)) -> 5)
                )
            else:
                coefficients.append(1)
                operand_map[arg] = arg  # Map to self
            operands.append(arg)

        try:
            # Map operands to values array with coefficients
            for operand, coeff in zip(operands, coefficients):
                operand_idx = values.index(
                    operand_map[operand]
                )  # Find index of this operand
                O[0, 0, step_index, operand_idx] += coeff  # Accumulate coefficients
        except Exception:
            print(f"Error when processing {node}")
            print(f"operand_map: {operand_map}")
            print(f"operands: {operands}")
            print(f"coefficients: {coefficients}")
            print(f"values: {values}")
            print(f"step_index: {step_index}")
            print(f"dag_depth: {dag_depth}")
            raise

        values.append(node)
        step_index += 1

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

    # This is for one batch element and one token element.
    # Hence the [0, 0, ...] indices.
    return target_digits, V_sign, O, G


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
    # Get tensor dimensions
    num_initial_nodes = digit_logits.shape[0]
    dag_depth = O.shape[0]

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
                    result_expr = sympy.Mul(-1, operand, evaluate=False)
                else:
                    result_expr = sympy.Mul(coeff, operand, evaluate=False)
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
                # Use unevaluated operations to preserve expression structure
                if domain_gate == 0:  # Log domain (Mul/Div)
                    if abs(coeff1 - 1) < 1e-10 and abs(coeff2 - 1) < 1e-10:
                        result_expr = sympy.Mul(operand1, operand2, evaluate=False)
                    elif abs(coeff1 - 1) < 1e-10 and abs(coeff2 + 1) < 1e-10:
                        result_expr = sympy.Mul(
                            operand1,
                            sympy.Pow(operand2, -1, evaluate=False),
                            evaluate=False,
                        )
                    else:
                        # General case with coefficients
                        if coeff2 > 0:
                            result_expr = sympy.Mul(
                                operand1,
                                sympy.Pow(operand2, coeff2, evaluate=False),
                                evaluate=False,
                            )
                        else:
                            result_expr = sympy.Mul(
                                operand1,
                                sympy.Pow(operand2, -abs(coeff2), evaluate=False),
                                evaluate=False,
                            )
                else:  # Linear domain (Add/Sub)
                    if abs(coeff1 - 1) < 1e-10 and abs(coeff2 - 1) < 1e-10:
                        result_expr = sympy.Add(operand1, operand2, evaluate=False)
                    elif abs(coeff1 - 1) < 1e-10 and abs(coeff2 + 1) < 1e-10:
                        result_expr = sympy.Add(
                            operand1,
                            sympy.Mul(-1, operand2, evaluate=False),
                            evaluate=False,
                        )
                    else:
                        # General case with coefficients
                        result_expr = sympy.Add(
                            sympy.Mul(coeff1, operand1, evaluate=False),
                            sympy.Mul(coeff2, operand2, evaluate=False),
                            evaluate=False,
                        )
            elif len(valid_operands) == 1:
                # Only one valid operand - treat as unary
                operand, coeff = valid_operands[0]
                if abs(coeff + 1) < 1e-10:
                    result_expr = sympy.Mul(-1, operand, evaluate=False)
                else:
                    result_expr = sympy.Mul(coeff, operand, evaluate=False)
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
                    terms.append(sympy.Mul(-1, operand, evaluate=False))
                else:
                    terms.append(sympy.Mul(coeff, operand, evaluate=False))

            if len(terms) == 0:
                # No valid operands, use first initial value as fallback
                result_expr = (
                    node_expressions[0] if node_expressions else sympy.Integer(0)
                )
            elif len(terms) == 1:
                result_expr = terms[0]
            elif domain_gate == 0:  # Log domain - multiply
                result_expr = sympy.Mul(*terms, evaluate=False)
            else:  # Linear domain - add
                result_expr = sympy.Add(*terms, evaluate=False)

        # Store intermediate result
        if len(node_expressions) <= num_initial_nodes + step:
            node_expressions.append(result_expr)
        else:
            node_expressions[num_initial_nodes + step] = result_expr

    # Return the final expression (last intermediate result or first value if no operations)
    if dag_depth > 0 and len(node_expressions) > num_initial_nodes:
        return node_expressions[-1]
    return node_expressions[0] if node_expressions else sympy.Integer(0)


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
        target_digits_shape = self.structure_dict["target_digits"].shape
        V_sign_shape = self.structure_dict["target_V_sign"].shape
        O_shape = self.structure_dict["target_O"].shape
        G_shape = self.structure_dict["target_G"].shape
        final_value_exec = self.structure_dict["target_final_exec"]
        return f"DAGValExample(seed={self.seed}, text={self.text}, depth={self.depth}, target_digits={target_digits_shape}, V_sign={V_sign_shape}, O={O_shape}, G={G_shape}, full_operations_named={self.full_operations_named}, final_value_sympy={self.final_value_sympy}, final_value_exec={final_value_exec}, full_expr={self.full_expr})"


def expressions_to_tensors(
    expressions: list[sympy.Basic | str],
    max_tokens: int,
    *,
    depth: int,
    max_digits: int = 4,
    max_decimal_places: int = 4,
) -> tuple[list[dict[str, torch.Tensor]], list[bool]]:
    """Convert a list of sympy expressions to structure tensors for the digit prediction system. Will pad with zero DAGs to max_tokens.

    This converts SymPy expressions to DAG tensor format:
    - Uses digit prediction, V_sign, O, G tensor representation
    - Invalid expressions get "zero DAG" representations

    Args:
        expressions: List of sympy expressions or "not valid" strings
        depth: Target depth for DAG operations
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places

    Returns:
        Dictionary with tensors for each target type, and a valid mask.
        The tensors are stacked along the token dimension (1).
        The valid mask is a boolean tensor of shape (B, T) indicating which token positions are valid.
    """

    if max_tokens < len(expressions):
        raise ValueError(
            f"max_tokens must be greater than or equal to the number of expressions: {max_tokens} < {len(expressions)}"
        )
    total_nodes = (depth + 1) + depth

    valid_mask_list = []
    target_digits_list = []
    target_V_sign_list = []
    target_O_list = []
    target_G_list = []
    target_final_exec_list = []

    base = 10  # Default base
    # Zero DAG for invalid token position
    target_digits_invalid = torch.zeros(
        1, 1, depth + 1, max_digits + max_decimal_places, base
    )
    target_digits_invalid[:, :, 0, 0] = 1.0
    target_V_sign_invalid = torch.ones(1, 1, total_nodes)
    target_O_invalid = torch.zeros(1, 1, depth, total_nodes)
    target_G_invalid = torch.ones(1, 1, depth)
    valid_mask_invalid = torch.zeros(1, 1, dtype=torch.bool)
    target_final_exec_invalid = torch.zeros(1, 1)

    for i in range(max_tokens):
        try:
            target_digits = target_digits_invalid
            target_V_sign = target_V_sign_invalid
            target_O = target_O_invalid
            target_G = target_G_invalid
            target_final_exec = target_final_exec_invalid
            valid_mask = valid_mask_invalid

            if i < len(expressions) and expressions[i] != "not valid":
                expr = expressions[i]
                target_digits, V_sign, target_O, target_G = expression_to_tensors(
                    expr, depth, max_digits, max_decimal_places
                )
                target_final_exec_value = float(expr.evalf())
                valid_mask = torch.ones(1, 1, dtype=torch.bool)

            # Assign to lists
            target_digits_list.append(target_digits)
            target_V_sign_list.append(V_sign)
            target_O_list.append(target_O)
            target_G_list.append(target_G)
            target_final_exec_list.append(
                torch.tensor(target_final_exec_value).unsqueeze(0)
            )
            valid_mask_list.append(valid_mask)

        except:
            string_expr = expressions[i] if i < len(expressions) else "not valid"
            print(f"Conversion failed for expression: {string_expr}")
            string_expr2 = str(sympy.sympify(string_expr, evaluate=False))
            print(f"Other format: {string_expr2}")
            string_expr3 = str(sympy.sympify(string_expr2, evaluate=False))
            print(f"Other format: {string_expr3}")
            raise

    # Stack the lists on token dimension (1)
    target_digits = torch.stack(target_digits_list, dim=1)
    target_V_sign = torch.stack(target_V_sign_list, dim=1)
    target_O = torch.stack(target_O_list, dim=1)
    target_G = torch.stack(target_G_list, dim=1)
    target_final_exec = torch.stack(target_final_exec_list, dim=1)
    valid_mask = torch.stack(valid_mask_list, dim=1)
    total_expressions = torch.tensor(len(expressions)).unsqueeze(0)

    tensors = {
        "target_digits": target_digits,
        "target_V_sign": target_V_sign,
        "target_O": target_O,
        "target_G": target_G,
        "valid_mask": valid_mask,
        "target_final_exec": target_final_exec,
        "total_expressions": total_expressions,
    }

    return tensors


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
    """

    def generate_batch(batch_size: int, training_seed: int):
        """Generate a batch using the per-token format."""
        tokenizer = get_encoding("gpt2")

        all_texts = []
        all_target_tensors = []

        for i in range(batch_size):
            # Generate expressions
            expressions, substrings, _ = generate_expressions(
                depth=max_depth,
                seed=training_seed + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=tokenizer,
            )

            # Convert to tensors
            target_tensors = expressions_to_tensors(
                expressions,
                depth=max_depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                max_tokens=block_size,
            )

            # Always use the full expression substring to maintain alignment with targets
            text = substrings[-1]
            all_texts.append(text)
            all_target_tensors.append(target_tensors)

        # Stack target tensors along batch dimension (dim=0)
        batched_target_tensors = {}
        for key in target_tensors.keys():
            tensors = [tensor[key] for tensor in all_target_tensors]
            batched_target_tensors[key] = torch.stack(tensors, dim=0)
        return all_texts, batched_target_tensors

    def train_loader():
        counter = 0
        while True:
            texts, target_tensors = generate_batch(train_batch_size, seed + counter)
            counter += train_batch_size
            yield texts, target_tensors

    def val_loader():
        counter = 0
        while True:
            texts, target_tensors = generate_batch(
                val_batch_size, seed + counter + 10000
            )
            counter += val_batch_size
            yield texts, target_tensors

    return train_loader(), val_loader()
