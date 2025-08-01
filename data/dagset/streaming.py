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


class Neg(sympy.Function):
    """Custom negation operation: Neg(x) = -x"""

    @classmethod
    def eval(cls, arg):
        """Don't evaluate - keep as symbolic custom operation"""
        return None  # Return None to prevent evaluation

    def evalf(self, *args, **kwargs):
        """Evaluate numerically when needed"""
        return -self.args[0].evalf(*args, **kwargs)


def normalize_expression(expr: sympy.Basic) -> sympy.Basic:
    """
    Normalize a SymPy expression by converting complex patterns into custom operations.

    Transformations:
    - Mul(a, Pow(b, -1), Pow(c, -1)) → Div(a, b, c)  # a * (1/b) * (1/c) → a / b / c
    - Add(a, Mul(-1, b), Mul(-1, c)) → Sub(a, b, c)  # a + (-b) + (-c) → a - b - c
    - Mul(-1, x) → Neg(x)  # -1 * x → -x
    """

    def transform_node(node):
        if isinstance(node, sympy.Mul):
            # First check if this is a negation: -1 * x
            if len(node.args) == 2 and node.args[0] == -1:
                # This is -1 * x, transform to Neg(x)
                return Neg(transform_node(node.args[1]))

            # Check if this is a division: a * (1/b) * (1/c) * ...
            numerators = []
            denominators = []

            for arg in node.args:
                if isinstance(arg, sympy.Pow) and arg.exp == -1:
                    # This is 1/x, so x is a denominator
                    denominators.append(transform_node(arg.base))
                else:
                    # Regular term, so it's a numerator
                    numerators.append(transform_node(arg))

            if denominators:  # Has reciprocals, so it's a division
                if len(numerators) == 1:
                    return Div(numerators[0], *denominators)
                else:
                    # Multiple numerators: (a*b) / c / d
                    combined_numerator = sympy.Mul(*numerators, evaluate=False)
                    return Div(combined_numerator, *denominators)
            else:
                # No reciprocals, just a regular multiplication
                if len(node.args) == 1:
                    return transform_node(node.args[0])
                else:
                    return sympy.Mul(
                        *[transform_node(arg) for arg in node.args], evaluate=False
                    )

        elif isinstance(node, sympy.Add):
            positive_terms = []
            negative_terms = []

            for arg in node.args:
                transformed_arg = transform_node(arg)
                if isinstance(transformed_arg, Neg):
                    # This is a Neg operation, extract its argument
                    negative_terms.append(transformed_arg.args[0])
                elif (
                    isinstance(arg, sympy.Mul)
                    and len(arg.args) == 2
                    and arg.args[0] == -1
                ):
                    # This is -1 * something, extract the something
                    negative_terms.append(transform_node(arg.args[1]))
                else:
                    positive_terms.append(transformed_arg)

            if negative_terms:  # Has negative terms, so it's a subtraction
                if len(positive_terms) == 1:
                    return Sub(positive_terms[0], *negative_terms)
                else:
                    # Multiple positive terms: (a+b) - c - d
                    combined_positive = sympy.Add(*positive_terms, evaluate=False)
                    return Sub(combined_positive, *negative_terms)
            else:
                # No negative terms, just a regular addition
                if len(node.args) == 1:
                    return transform_node(node.args[0])
                else:
                    return sympy.Add(
                        *[transform_node(arg) for arg in node.args], evaluate=False
                    )

        elif isinstance(node, sympy.Pow):
            # Convert Pow(x, -1) to Div(1, x)
            if node.exp == -1:
                return Div(sympy.Integer(1), transform_node(node.base))
            else:
                # Other powers are passed through (but will error in expression_to_tensors)
                return sympy.Pow(transform_node(node.base), node.exp)

        elif isinstance(node, (sympy.Symbol, sympy.Integer, sympy_float)):
            # Leaf nodes are returned as-is
            return node

        else:
            # Other node types (functions, etc.) - recursively transform arguments
            if hasattr(node, "args") and node.args:
                return node.func(*[transform_node(arg) for arg in node.args])
            else:
                return node

    return transform_node(expr)


def is_negation(node: sympy.Basic) -> bool:
    """Check if a node represents -1 * value (SymPy artifact that should have been normalized)."""
    if not isinstance(node, sympy.Mul) or len(node.args) != 2:
        return False

    first_arg, second_arg = node.args

    # Only flag the specific pattern we try to normalize: -1 * x (first arg is -1)
    return (first_arg == -1) and (
        isinstance(second_arg, sympy_float) or isinstance(second_arg, sympy.Integer)
    )


def expression_to_tensors(
    expr: sympy.Basic, dag_depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a SymPy expression to DAG tensor representation.

    Returns:
        V_mag: (1, 1, dag_depth * 2) - magnitudes of all nodes (initial + intermediate)
        V_sign: (1, 1, dag_depth * 2) - signs of all nodes
        O: (1, 1, dag_depth, dag_depth * 2) - operand selection matrix
        G: (1, 1, dag_depth) - domain selector (0=log, 1=linear)
    """
    # Step 1: Normalize the expression
    normalized_expr = normalize_expression(expr)

    # Initialize tensors with dag_depth * 2 architecture (matching newdag.py)
    V_mag = torch.zeros(1, 1, dag_depth * 2)
    V_sign = torch.ones(1, 1, dag_depth * 2)  # Initialize to all 1s
    O = torch.zeros(1, 1, dag_depth, dag_depth * 2)
    G = torch.ones(1, 1, dag_depth)  # Initialize to all 1s (linear domain)

    # Track values list and step index
    values = []
    step_index = 0
    node_replacements = {}  # For Neg of numbers

    # PHASE 1: Collect all initial numeric values (including Neg of numbers)
    for node in postorder_traversal(normalized_expr):
        is_float = isinstance(node, sympy_float)
        is_int = isinstance(node, sympy.Integer)

        # Handle direct numbers
        if is_float or is_int:
            if step_index >= dag_depth:
                raise ValueError(
                    f"Expression has too many initial values. Needs dag_depth > {step_index}"
                )

            try:
                val = float(node.evalf())
                V_mag[0, 0, step_index] = abs(val)
                V_sign[0, 0, step_index] = 1 if val >= 0 else -1
                values.append(node)
                step_index += 1
            except (TypeError, ValueError):
                raise ValueError(f"Cannot convert {node} to float")

        # Handle Neg of numbers (like Neg(5) = -5)
        elif isinstance(node, Neg):
            arg = node.args[0]
            if isinstance(arg, (sympy_float, sympy.Integer)):
                if step_index >= dag_depth:
                    raise ValueError(
                        f"Expression has too many initial values. Needs dag_depth > {step_index}"
                    )

                try:
                    val = float(arg.evalf())
                    # Negate the value for Neg operation
                    V_mag[0, 0, step_index] = abs(val)
                    V_sign[0, 0, step_index] = -1 if val >= 0 else 1  # Flip sign

                    # Create a replacement node for this negated value
                    negated_node = (
                        sympy_float(-val)
                        if isinstance(arg, sympy_float)
                        else sympy.Integer(-int(val))
                    )
                    values.append(negated_node)
                    node_replacements[node] = negated_node
                    step_index += 1
                except (TypeError, ValueError):
                    raise ValueError(f"Cannot convert Neg({arg}) to float")

        # Error checks (moved to Phase 1)
        elif is_negation(node):
            raise ValueError(
                f"Unexpected old-style negation after normalization: {node}. "
                f"All Mul(-1, x) patterns should have been converted to Neg(x) during normalization. "
                f"Normalization failed to catch this pattern."
            )
        elif isinstance(node, sympy.Pow):
            raise ValueError(
                f"Unexpected Pow node after normalization: {node}. "
                f"All Pow(x, -1) should be converted to Div(1, x). "
                f"Other Pow operations are not supported."
            )

    # Pad values list to dag_depth
    while len(values) < dag_depth:
        values.append("UNUSED")

    # Reset step_index for operations
    step_index = 0

    # PHASE 2: Process operations in second postorder traversal
    for node in postorder_traversal(normalized_expr):
        # Skip leaf nodes (already processed in Phase 1)
        is_float = isinstance(node, sympy_float)
        is_int = isinstance(node, sympy.Integer)
        if is_float or is_int:
            continue

        # Skip Neg operations for numbers (already processed in Phase 1)
        if isinstance(node, Neg):
            arg = node.args[0]
            if isinstance(arg, (sympy_float, sympy.Integer)):
                continue  # Already handled in Phase 1

        if step_index >= dag_depth:
            raise ValueError(
                f"Expression requires more operations than dag_depth={dag_depth}"
            )

        is_neg = isinstance(node, Neg)
        is_div = isinstance(node, Div)
        is_sub = isinstance(node, Sub)
        is_mul = isinstance(node, sympy.Mul)
        is_add = isinstance(node, sympy.Add)

        # Set domain: log for Div/Mul, linear for Add/Sub/Neg
        if is_div or is_mul:
            G[0, 0, step_index] = 0  # Log domain
        elif is_add or is_sub or is_neg:
            G[0, 0, step_index] = 1  # Linear domain

        # Determine coefficients based on operation type
        if is_neg:
            first_coefficient = -1
            # coefficient is unused for Neg (as it's unary)
        elif is_div or is_sub:
            first_coefficient = 1
            coefficient = -1
        elif is_add or is_mul:
            first_coefficient = 1
            coefficient = 1

        # Process arguments and set operand matrix
        if is_div or is_sub or is_add or is_mul or is_neg:
            for i, arg in enumerate(node.args):
                arg = node_replacements.get(arg, arg)  # Use replacement if exists
                arg_idx = values.index(arg)  # Find index of this argument
                _coefficient = first_coefficient if i == 0 else coefficient
                O[0, 0, step_index, arg_idx] += _coefficient  # Accumulate coefficients

            values.append(node)
            step_index += 1
        else:
            raise ValueError(
                f"Unexpected node type: {type(node)}"
            )  # Defensive programming

    # Fill remaining steps with identity operations (matching newdag.py)
    if step_index > 0:
        # Had operations - use the last operation result slot
        # The executor stores operation results at dag_depth + step_number
        last_result_slot = dag_depth + (step_index - 1)
        for remaining_step in range(step_index, dag_depth):
            O[0, 0, remaining_step, last_result_slot] = 1
    else:
        # No operations - use the first initial value
        for remaining_step in range(0, dag_depth):
            O[0, 0, remaining_step, 0] = 1

    return V_mag, V_sign, O, G


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


# ============================================================================
# EXTRACTION SYSTEM: Convert GENERATION_OPS to STATE_OPS
# ============================================================================


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
) -> tuple[list[dict[str, torch.Tensor]], list[bool]]:
    """Convert a list of sympy expressions to structure tensors for the new DAG system.

    This converts SymPy expressions to DAG tensor format:
    - Uses V_mag, V_sign, O, G tensor representation
    - Invalid expressions get "zero DAG" representations

    Args:
        expressions: List of sympy expressions or "not valid" strings
        depth: Target depth for DAG operations

    Returns:
        Tuple of (tensor_list, valid_mask) where:
        - tensor_list: Contains T tensors (one per token position, including zero DAGs)
        - valid_mask: Boolean list indicating which positions were valid
    """
    tensor_results = []
    valid_mask = []

    for expr in expressions:
        if expr == "not valid":
            # Create zero DAG for invalid token position
            zero_tensor_dict = {
                "target_V_mag": torch.zeros(depth * 2),
                "target_V_sign": torch.ones(depth * 2),
                "target_O": torch.zeros(depth, depth * 2),
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

                # Convert to target format for training
                # Remove batch and time dimensions (convert from (1,1,X) to (X))
                target_dict = {
                    "target_V_mag": V_mag.squeeze(0).squeeze(0),  # (total_nodes,)
                    "target_V_sign": V_sign.squeeze(0).squeeze(0),  # (total_nodes,)
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
            expressions, substrings, valid_mask_list = generate_expression(
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
            zero_tensor_dict = {
                "target_V_mag": torch.zeros(max_depth * 2),
                "target_V_sign": torch.ones(max_depth * 2),
                "target_O": torch.zeros(max_depth, max_depth * 2),
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
