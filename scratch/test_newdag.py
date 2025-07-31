"""
Test implementation of the new DAG design with domain mixing.

Key innovations:
- V_mag: absolute values in linear space
- V_sign: ±1 signs
- O: signed operand mask (negative weights = subtraction/division)
- S: domain selector (0=log/mul, 1=linear/add)
- Domain mixing: mixed = log(V_mag) * (1-S) + V_mag * S
"""

from typing import Tuple

import sympy
import torch
import torch.nn as nn

# Stability constants from dag_model.py
LOG_LIM = 23.026  # Bound on ln-magnitudes (≈ 10.0 * ln(10))
MIN_CLAMP = 1e-6


def _clip_log(log_t: torch.Tensor) -> torch.Tensor:
    """Symmetric clipping for log magnitudes to prevent numerical instabilities."""
    needs_clipping = torch.abs(log_t) > (0.9 * LOG_LIM)
    clipped = torch.tanh(log_t / LOG_LIM) * LOG_LIM
    return torch.where(needs_clipping, clipped, log_t)


def safe_log(x: torch.Tensor, eps: float = MIN_CLAMP) -> torch.Tensor:
    """Safe logarithm with clipping and precision preservation."""
    # Use double precision for small numbers to minimize error
    target_dtype = torch.float64 if x.device.type != "mps" else torch.float32

    # Only clamp very small values, preserving precision for normal values
    x_clamped = torch.clamp(x, min=eps)

    # For very small values, use double precision
    needs_precision = x < 1e-3
    if needs_precision.any():
        x_precise = x_clamped.to(target_dtype)
        log_precise = torch.log(x_precise).to(x.dtype)
        log_normal = torch.log(x_clamped)
        log_result = torch.where(needs_precision, log_precise, log_normal)
    else:
        log_result = torch.log(x_clamped)

    return _clip_log(log_result)


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Safe exponential with clipping and precision preservation."""
    clipped_x = torch.clamp(x, max=LOG_LIM)

    # Use double precision for values that might produce very small results
    target_dtype = torch.float64 if x.device.type != "mps" else torch.float32
    needs_precision = clipped_x < -10.0  # exp(-10) ≈ 4.5e-5

    if needs_precision.any():
        x_precise = clipped_x.to(target_dtype)
        exp_precise = torch.exp(x_precise).to(x.dtype)
        exp_normal = torch.exp(clipped_x)
        return torch.where(needs_precision, exp_precise, exp_normal)
    else:
        return torch.exp(clipped_x)


def smooth_sign(x: torch.Tensor, temperature: float = 0.0001) -> torch.Tensor:
    """Smooth approximation to sign function using tanh with adaptive temperature."""
    # Use smaller temperature for better approximation to true sign function
    # For small values, we want tanh to saturate quickly to ±1
    return torch.tanh(x / temperature)


class NewDAGPlanPredictor(nn.Module):
    """
    Predict complete DAG execution plan upfront (following dag_model.py pattern).

    Structure: 50/50 split of total nodes
    - initial_slots = dag_depth // 2 (for input values)
    - intermediate_slots = dag_depth // 2 (for computation results)
    - total_nodes = dag_depth

    Predicts:
    - Initial values V_mag, V_sign for all total_nodes
    - All operand selectors O for all steps: (B, T, intermediate_slots, total_nodes)
    - All domain gates G for all steps: (B, T, intermediate_slots)
    """

    def __init__(self, dag_depth: int, hidden_dim: int = 512):
        super().__init__()
        self.dag_depth = dag_depth
        self.total_nodes = dag_depth
        self.initial_slots = dag_depth // 2
        self.intermediate_slots = dag_depth // 2
        self.hidden_dim = hidden_dim

        # Predictors for complete plan (like DAGPlanPredictor in dag_model.py)
        self.initial_values_predictor = nn.Linear(
            hidden_dim, self.total_nodes * 2
        )  # mag + sign for all nodes
        self.operand_selectors_predictor = nn.Linear(
            hidden_dim, self.intermediate_slots * self.total_nodes
        )  # O for all steps
        self.domain_gates_predictor = nn.Linear(
            hidden_dim, self.intermediate_slots
        )  # G for all steps

        # Initialize with reasonable weights
        nn.init.xavier_uniform_(self.initial_values_predictor.weight)
        nn.init.xavier_uniform_(self.operand_selectors_predictor.weight)
        nn.init.xavier_uniform_(self.domain_gates_predictor.weight)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict complete DAG execution plan with 50/50 split.

        Args:
            hidden: (B, T, H) hidden states

        Returns:
            V_mag: (B, T, total_nodes) magnitudes for all nodes (initial + intermediate slots)
            V_sign: (B, T, total_nodes) signs for all nodes
            O: (B, T, intermediate_slots, total_nodes) operand selectors for each computation step
            G: (B, T, intermediate_slots) domain gates for each computation step
        """
        B, T, H = hidden.shape

        # Predict values for all nodes (initial values go in first half, intermediate results in second half)
        initial_raw = self.initial_values_predictor(hidden)  # (B, T, total_nodes * 2)
        V_mag = torch.abs(
            initial_raw[..., : self.total_nodes]
        )  # (B, T, total_nodes) - always positive
        V_sign = torch.tanh(
            initial_raw[..., self.total_nodes :]
        )  # (B, T, total_nodes) - in [-1, 1]

        # Predict operand selectors for intermediate computation steps
        operand_raw = self.operand_selectors_predictor(
            hidden
        )  # (B, T, intermediate_slots * total_nodes)
        O = operand_raw.view(
            B, T, self.intermediate_slots, self.total_nodes
        )  # (B, T, intermediate_slots, total_nodes)

        # Predict domain gates for intermediate computation steps
        domain_raw = self.domain_gates_predictor(hidden)  # (B, T, intermediate_slots)
        G = torch.sigmoid(domain_raw)  # (B, T, intermediate_slots) - in [0, 1]

        return V_mag, V_sign, O, G


class NewDAGExecutor(nn.Module):
    """
    New DAG executor with domain mixing approach and 50/50 node split.
    Executes using pre-predicted complete plan (like execute_stack in dag_model.py).

    Structure:
    - total_nodes = dag_depth
    - initial_slots = dag_depth // 2 (positions 0 to initial_slots-1)
    - intermediate_slots = dag_depth // 2 (positions initial_slots to total_nodes-1)
    """

    def __init__(self, dag_depth: int, hidden_dim: int = 512):
        super().__init__()
        self.dag_depth = dag_depth
        self.total_nodes = dag_depth
        self.initial_slots = dag_depth // 2
        self.intermediate_slots = dag_depth // 2
        self.hidden_dim = hidden_dim

        # Plan predictor (like DAGPlanPredictor in dag_model.py)
        self.plan_predictor = NewDAGPlanPredictor(dag_depth, hidden_dim)

    def execute_with_plan(
        self,
        V_mag: torch.Tensor,  # (B, T, total_nodes) all node magnitudes
        V_sign: torch.Tensor,  # (B, T, total_nodes) all node signs
        O: torch.Tensor,  # (B, T, intermediate_slots, total_nodes) operand selectors
        G: torch.Tensor,  # (B, T, intermediate_slots) domain gates
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Execute DAG using complete pre-predicted plan with fixed-size tensors.

        Structure:
        - V_mag[:,:,0:initial_slots] = initial values (predicted/given)
        - V_mag[:,:,initial_slots:] = intermediate results (computed during execution)

        Args:
            V_mag: All node magnitudes (initial + intermediate slots)
            V_sign: All node signs (initial + intermediate slots)
            O: Operand selectors for intermediate computation steps
            G: Domain gates for intermediate computation steps
            debug: Whether to print debug info

        Returns:
            final_values: (B, T) final computed values
        """
        B, T, total_nodes = V_mag.shape

        # Work with a copy to avoid modifying the input
        working_V_mag = V_mag.clone()  # (B, T, total_nodes)
        working_V_sign = V_sign.clone()  # (B, T, total_nodes)

        if debug:
            print(f"Fixed-size execution:")
            print(f"  Total nodes: {total_nodes}")
            print(
                f"  Initial slots: {self.initial_slots} (positions 0:{self.initial_slots})"
            )
            print(
                f"  Intermediate slots: {self.intermediate_slots} (positions {self.initial_slots}:{total_nodes})"
            )
            print(f"  V_mag shape: {working_V_mag.shape}")
            print(f"  V_sign shape: {working_V_sign.shape}")
            print(f"  O shape: {O.shape}")
            print(f"  G shape: {G.shape}")
            print()

        # Execute each intermediate computation step
        for step in range(self.intermediate_slots):
            # operand selector and domain gate for this step
            O_step = O[:, :, step, :]  # (B, T, total_nodes)
            G_step = G[:, :, step].unsqueeze(-1)  # (B, T, 1)

            # Apply triangular mask to prevent using future intermediate results
            valid_positions = (
                self.initial_slots + step
            )  # How many positions are available

            # Create causal mask
            causal_mask = torch.zeros_like(O_step)  # (B, T, total_nodes)
            causal_mask[:, :, :valid_positions] = 1.0

            # Apply mask to operand selector
            O_step = O_step * causal_mask

            if debug:
                print(
                    f"Step {step}: Can use positions 0:{valid_positions} (out of {total_nodes})"
                )
                n_show = min(8, total_nodes)
                print(
                    f"  O_step (before mask): {O[:, :, step, :n_show][0, 0].tolist()}"
                )
                print(f"  O_step (after mask): {O_step[0, 0, :n_show].tolist()}")

            # All DAG computations in double precision for maximum numerical stability
            # Performance impact is negligible since DAG execution << LLM inference
            target_dtype = (
                torch.float64 if working_V_mag.device.type != "mps" else torch.float32
            )

            # Convert to double precision
            working_V_mag_hp = working_V_mag.to(target_dtype)
            working_V_sign_hp = working_V_sign.to(target_dtype)
            O_step_hp = O_step.to(target_dtype)
            G_step_hp = G_step.to(target_dtype)

            signed_values = working_V_sign_hp * working_V_mag_hp
            log_mag_hp = torch.log(torch.clamp(working_V_mag_hp, min=1e-12))
            mixed = log_mag_hp * (1 - G_step_hp) + signed_values * G_step_hp
            R_mag = torch.sum(O_step_hp * mixed, dim=-1, keepdim=True)

            linear_sign = torch.tanh(R_mag / 0.0001)  # Direct tanh for better precision

            # Log domain sign: product of selected signs
            sign_weights = working_V_sign_hp * (
                torch.abs(O_step_hp) + 1
            )  # +1 for unselected
            sign_product = torch.prod(sign_weights, dim=-1, keepdim=True)
            log_sign = torch.tanh(sign_product / 0.0001)

            V_sign_new = G_step_hp * linear_sign + (1 - G_step_hp) * log_sign

            # For magnitude: linear domain uses abs(R_mag), log domain uses exp(R_mag)
            linear_mag = torch.abs(R_mag)
            log_mag_result = torch.exp(
                torch.clamp(R_mag, max=23.026)
            )  # Use LOG_LIM for safety
            V_mag_new = G_step_hp * linear_mag + (1 - G_step_hp) * log_mag_result

            # Convert back to original precision
            V_sign_new = V_sign_new.to(working_V_mag.dtype)
            V_mag_new = V_mag_new.to(working_V_mag.dtype)

            # Write result to the predetermined intermediate slot (functional update for gradient flow)
            intermediate_idx = self.initial_slots + step
            # Use scatter to avoid in-place operations that break gradients
            indices = torch.tensor(
                [intermediate_idx], device=working_V_mag.device
            ).expand(working_V_mag.shape[:2])
            working_V_mag = working_V_mag.scatter(-1, indices.unsqueeze(-1), V_mag_new)
            working_V_sign = working_V_sign.scatter(
                -1, indices.unsqueeze(-1), V_sign_new
            )

            if debug:
                result_value = V_sign_new * V_mag_new
                print(f"  G_step: {G_step[0, 0, 0].item():.3f}")
                print(
                    f"  -> slot {intermediate_idx}: {result_value[0, 0, 0].item():.3f}"
                )
                print()

        # Always use the last intermediate slot as the final result, models should use identity operation for the rest once they are done with the computation
        final_idx = total_nodes - 1
        final_mag = working_V_mag[:, :, final_idx]  # (B, T)
        final_sign = working_V_sign[:, :, final_idx]  # (B, T)
        final_value = final_sign * final_mag  # (B, T)

        if debug:
            print(f"Final result from slot {final_idx}: {final_value}")

        return final_value

    def forward(self, hidden: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Full forward pass: predict complete plan, then execute.

        Args:
            hidden: (B, T, H) hidden states
            debug: Whether to print debug info

        Returns:
            final_values: (B, T) final computed values
        """
        # Predict complete plan upfront (like DAGPlanPredictor)
        V_mag, V_sign, O, G = self.plan_predictor(hidden)

        # Execute using the complete plan (like execute_stack)
        return self.execute_with_plan(V_mag, V_sign, O, G, debug=debug)


def expression_to_tensors(
    expr: sympy.Basic, dag_depth: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a sympy expression to predictor tensor format using simplified stack-based approach.

    Args:
        expr: Sympy expression to convert
        dag_depth: Fixed depth for tensor allocation

    Returns:
        V_mag, V_sign, O, G: Tensors for DAG execution
    """
    total_nodes = dag_depth
    initial_slots = dag_depth // 2
    intermediate_slots = dag_depth // 2

    # Phase 1: Extract symbols and constants simply
    symbols = sorted(expr.free_symbols, key=str)
    constants = []

    # Find all numeric constants in the expression
    def find_constants(node):
        if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational)):
            constants.append(float(node))
        elif isinstance(node, sympy.Pow):
            # For power nodes like b**(-1), only collect constants from the base, not the exponent
            find_constants(node.args[0])  # Only process the base
        elif hasattr(node, "args"):
            for arg in node.args:
                find_constants(arg)

    find_constants(expr)
    constants = list(dict.fromkeys(constants))  # Remove duplicates, preserve order

    # Build initial values: symbols first, then constants
    initial_values = [1.0] * len(symbols) + constants
    initial_values = (initial_values + [1.0] * initial_slots)[:initial_slots]

    # Create symbol/constant lookup
    atom_to_slot = {}
    for i, sym in enumerate(symbols[:initial_slots]):
        atom_to_slot[sym] = i
    for i, const in enumerate(constants):
        if len(symbols) + i < initial_slots:
            atom_to_slot[const] = len(symbols) + i

    # Phase 2: Simple post-order traversal to build operations
    operations = []  # (operand_weights, is_linear_domain)
    node_to_slot = atom_to_slot.copy()
    next_slot = initial_slots

    def evaluate_node(node):
        """Post-order evaluation that builds operations."""
        nonlocal next_slot

        if node in node_to_slot:
            return node_to_slot[node]

        # Handle atomic values
        if isinstance(node, sympy.Symbol):
            return atom_to_slot.get(node, 0)
        elif isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational)):
            return atom_to_slot.get(float(node), 0)

        # Handle operations
        elif isinstance(node, sympy.Add):
            # Build operand weights (handle subtraction as negative weights)
            weights = [0.0] * total_nodes
            for arg in node.args:
                # Extract coefficient and base term
                if hasattr(arg, "as_coeff_Mul"):
                    coeff, rest = arg.as_coeff_Mul()
                    # Evaluate the base term (without coefficient)
                    slot = evaluate_node(rest)
                    if slot < total_nodes:
                        weights[slot] += float(coeff)
                else:
                    # Simple term without coefficient
                    slot = evaluate_node(arg)
                    if slot < total_nodes:
                        weights[slot] += 1.0

            operations.append((weights, True))  # Linear domain for addition
            node_to_slot[node] = next_slot
            next_slot += 1
            return node_to_slot[node]

        elif isinstance(node, sympy.Mul):
            # Process all operands first (post-order)
            operand_slots = [evaluate_node(arg) for arg in node.args]

            # Build operand weights (handle division as negative weights)
            weights = [0.0] * total_nodes
            for i, arg in enumerate(node.args):
                slot = operand_slots[i]
                if slot < total_nodes:
                    # Check if this is division (negative exponent)
                    if isinstance(arg, sympy.Pow) and arg.args[1] == -1:
                        weights[slot] -= 1.0  # Division in log domain
                    else:
                        weights[slot] += 1.0  # Multiplication in log domain

            operations.append((weights, False))  # Log domain for multiplication
            node_to_slot[node] = next_slot
            next_slot += 1
            return node_to_slot[node]

        elif isinstance(node, sympy.Pow) and node.args[1] == -1:
            # Handle 1/x as reciprocal of x
            return evaluate_node(node.args[0])

        else:
            # Fallback to first slot for unknown node types
            return 0

    # Evaluate the expression (this builds operations via post-order traversal)
    final_slot = evaluate_node(expr)

    # Phase 3: Build tensors directly
    V_mag = torch.zeros(1, 1, total_nodes)
    V_sign = torch.ones(1, 1, total_nodes)
    O = torch.zeros(1, 1, intermediate_slots, total_nodes)
    G = torch.zeros(1, 1, intermediate_slots)

    # Fill initial values
    for i, val in enumerate(initial_values):
        if i < total_nodes:
            V_mag[0, 0, i] = abs(val)
            V_sign[0, 0, i] = 1.0 if val >= 0 else -1.0

    # Fill operations
    for i, (weights, is_linear) in enumerate(operations[:intermediate_slots]):
        for j, weight in enumerate(weights):
            if j < total_nodes and abs(weight) > 1e-8:
                O[0, 0, i, j] = weight
        G[0, 0, i] = 1.0 if is_linear else 0.0

    # Fill remaining slots with identity operations
    for i in range(len(operations), intermediate_slots):
        source_slot = (
            final_slot if i == 0 and len(operations) == 0 else (initial_slots + i - 1)
        )
        if source_slot < total_nodes:
            O[0, 0, i, source_slot] = 1.0
            G[0, 0, i] = 1.0  # Linear domain for identity

    return V_mag, V_sign, O, G


def test_simple_arithmetic():
    """Test basic arithmetic operations with 50/50 split structure."""
    print("=== Testing Simple Arithmetic ===")

    # Create executor with depth=2 for 50/50 split: 1 initial slot + 1 intermediate slot
    executor = NewDAGExecutor(dag_depth=2, hidden_dim=64)

    # Test data: batch_size=1, sequence_length=1
    hidden_states = torch.randn(1, 1, 64)  # (B=1, T=1, H=64)

    # Test addition: 2 + 3 = 5
    print("Testing Addition (2 + 3):")
    print(
        f"Structure: {executor.initial_slots} initial slots + {executor.intermediate_slots} intermediate slots = {executor.total_nodes} total"
    )

    with torch.no_grad():
        # Set initial values: only position 0 is for initial values, position 1 is for intermediate result
        executor.plan_predictor.initial_values_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data.fill_(0.0)
        # We need to encode both operands in a single initial slot (this is a limitation of depth=2)
        # Let's use a different approach - use depth=4 for proper testing

    # Create better executor with depth=4: 2 initial + 2 intermediate
    executor = NewDAGExecutor(dag_depth=4, hidden_dim=64)
    print(
        f"Better structure: {executor.initial_slots} initial + {executor.intermediate_slots} intermediate = {executor.total_nodes} total"
    )

    with torch.no_grad():
        # Set initial values in positions 0,1
        executor.plan_predictor.initial_values_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data[0] = (
            2.0  # V_mag[0] = 2
        )
        executor.plan_predictor.initial_values_predictor.bias.data[1] = (
            3.0  # V_mag[1] = 3
        )
        executor.plan_predictor.initial_values_predictor.bias.data[4] = (
            1.0  # V_sign[0] = +1
        )
        executor.plan_predictor.initial_values_predictor.bias.data[5] = (
            1.0  # V_sign[1] = +1
        )

        # Set operand selector for step 0: select positions 0,1 (the two initial values)
        executor.plan_predictor.operand_selectors_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.operand_selectors_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.operand_selectors_predictor.bias.data[0] = (
            1.0  # O[0,0] = 1
        )
        executor.plan_predictor.operand_selectors_predictor.bias.data[1] = (
            1.0  # O[0,1] = 1
        )

        # Set domain gate for step 0: linear domain for addition
        executor.plan_predictor.domain_gates_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data[0] = (
            5.0  # sigmoid(5) ≈ 0.99
        )

    result = executor(hidden_states, debug=True)
    expected = 2.0 + 3.0
    print(f"Addition result: {result[0, 0].item():.3f} (expected ≈ {expected})")
    print()

    # Test multiplication: 2 * 3 = 6
    print("Testing Multiplication (2 * 3):")
    with torch.no_grad():
        # Keep same initial values and operand selectors
        # Only change domain gate to log domain for multiplication
        executor.plan_predictor.domain_gates_predictor.bias.data[0] = (
            -5.0
        )  # sigmoid(-5) ≈ 0.01

    result = executor(hidden_states, debug=True)
    expected = 2.0 * 3.0
    print(f"Multiplication result: {result[0, 0].item():.3f} (expected ≈ {expected})")
    print()


def test_subtraction_division():
    """Test subtraction and division using negative weights."""
    print("=== Testing Subtraction and Division ===")

    # Create executor with depth=4: 2 initial + 2 intermediate
    executor = NewDAGExecutor(dag_depth=4, hidden_dim=64)
    hidden_states = torch.randn(1, 1, 64)

    # Test subtraction: 5 - 2 = 3
    print("Testing Subtraction (5 - 2):")
    with torch.no_grad():
        # Set initial values in positions 0,1
        executor.plan_predictor.initial_values_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data[0] = (
            5.0  # V_mag[0] = 5
        )
        executor.plan_predictor.initial_values_predictor.bias.data[1] = (
            2.0  # V_mag[1] = 2
        )
        executor.plan_predictor.initial_values_predictor.bias.data[4] = (
            1.0  # V_sign[0] = +1
        )
        executor.plan_predictor.initial_values_predictor.bias.data[5] = (
            1.0  # V_sign[1] = +1
        )

        # Set operand selector for step 0: [1, -1, 0, 0] for subtraction
        executor.plan_predictor.operand_selectors_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.operand_selectors_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.operand_selectors_predictor.bias.data[0] = (
            1.0  # O[0,0] = 1
        )
        executor.plan_predictor.operand_selectors_predictor.bias.data[1] = (
            -1.0
        )  # O[0,1] = -1

        # Linear domain for subtraction
        executor.plan_predictor.domain_gates_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data[0] = (
            5.0  # sigmoid(5) ≈ 0.99
        )

    result = executor(hidden_states, debug=True)
    expected = 5.0 - 2.0
    print(f"Subtraction result: {result[0, 0].item():.3f} (expected = {expected})")
    print()

    # Test division: 5 / 2 = 2.5
    print("Testing Division (5 / 2):")
    with torch.no_grad():
        # Keep same initial values and operand selectors [1, -1]
        # Change to log domain: log(a) - log(b) = log(a/b)
        executor.plan_predictor.domain_gates_predictor.bias.data[0] = (
            -5.0
        )  # sigmoid(-5) ≈ 0.01

    result = executor(hidden_states, debug=True)
    expected = 5.0 / 2.0
    print(f"Division result: {result[0, 0].item():.3f} (expected = {expected})")
    print()


def test_complex_expression():
    """Test complex expression: (a + b) * (c + d) with 50/50 split and triangular masking."""
    print("=== Testing Complex Expression: (a + b) * (c + d) ===")

    # Create executor with depth=8: 4 initial + 4 intermediate (perfect for our example)
    executor = NewDAGExecutor(dag_depth=8, hidden_dim=64)
    hidden_states = torch.randn(1, 1, 64)  # (B=1, T=1, H=64)

    # Initial values: a=1, b=2, c=3, d=4
    # Expected: (1+2) * (3+4) = 3 * 7 = 21
    print("Testing: (1 + 2) * (3 + 4) = 21")
    print(
        f"Structure: {executor.initial_slots} initial + {executor.intermediate_slots} intermediate = {executor.total_nodes} total"
    )

    with torch.no_grad():
        # Set initial values in positions 0,1,2,3
        executor.plan_predictor.initial_values_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.initial_values_predictor.bias.data.fill_(0.0)
        # Magnitudes
        executor.plan_predictor.initial_values_predictor.bias.data[0] = 1.0  # a
        executor.plan_predictor.initial_values_predictor.bias.data[1] = 2.0  # b
        executor.plan_predictor.initial_values_predictor.bias.data[2] = 3.0  # c
        executor.plan_predictor.initial_values_predictor.bias.data[3] = 4.0  # d
        # Signs (all positive)
        for i in range(4):
            executor.plan_predictor.initial_values_predictor.bias.data[8 + i] = (
                1.0  # V_sign starts at position 8
            )

        # Set operand selectors for intermediate steps
        # O is shaped (B, T, intermediate_slots, total_nodes) -> (1, 1, 4, 8) -> flattened to (4 * 8) = 32
        executor.plan_predictor.operand_selectors_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.operand_selectors_predictor.bias.data.fill_(0.0)

        # Step 0: a + b (positions 0,1) -> result goes to position 4
        executor.plan_predictor.operand_selectors_predictor.bias.data[0] = (
            1.0  # O[0,0] = 1
        )
        executor.plan_predictor.operand_selectors_predictor.bias.data[1] = (
            1.0  # O[0,1] = 1
        )

        # Step 1: c + d (positions 2,3) -> result goes to position 5
        # Offset by total_nodes=8 for next step
        executor.plan_predictor.operand_selectors_predictor.bias.data[8 + 2] = (
            1.0  # O[1,2] = 1
        )
        executor.plan_predictor.operand_selectors_predictor.bias.data[8 + 3] = (
            1.0  # O[1,3] = 1
        )

        # Step 2: result1 * result2 (positions 4,5) -> result goes to position 6
        # Offset by 2*total_nodes=16 for step 2
        executor.plan_predictor.operand_selectors_predictor.bias.data[16 + 4] = (
            1.0  # O[2,4] = 1
        )
        executor.plan_predictor.operand_selectors_predictor.bias.data[16 + 5] = (
            1.0  # O[2,5] = 1
        )

        # Set domain gates for intermediate steps
        # G is shaped (B, T, intermediate_slots) -> (1, 1, 4) -> flattened to 4
        executor.plan_predictor.domain_gates_predictor.weight.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data.fill_(0.0)
        executor.plan_predictor.domain_gates_predictor.bias.data[0] = (
            5.0  # Step 0: linear (addition)
        )
        executor.plan_predictor.domain_gates_predictor.bias.data[1] = (
            5.0  # Step 1: linear (addition)
        )
        executor.plan_predictor.domain_gates_predictor.bias.data[2] = (
            -5.0
        )  # Step 2: log (multiplication)

    result = executor(hidden_states, debug=True)
    expected = (1 + 2) * (3 + 4)
    print(
        f"Complex expression result: {result[0, 0].item():.3f} (expected = {expected})"
    )
    print()


def test_complex_expressions_validation():
    """Test complex sympy expressions against DAG execution with high depth."""
    print("=== Testing Complex Expression Validation ===")

    # Test cases: (expression, variable_values)
    test_cases = [
        # Simple nested: (a + b) * (c - d)
        ("(a + b) * (c - d)", {"a": 2, "b": 3, "c": 7, "d": 2}),
        # Deep nesting: ((a + b) * c) + ((d - e) * f)
        (
            "((a + b) * c) + ((d - e) * f)",
            {"a": 1, "b": 2, "c": 3, "d": 5, "e": 1, "f": 2},
        ),
        # Mixed operations: (a * b + c) / (d - e)
        ("(a * b + c) / (d - e)", {"a": 2, "b": 3, "c": 1, "d": 5, "e": 2}),
        # Complex nested: (a + b * c) * (d + e / f)
        ("(a + b * c) * (d + e / f)", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 6, "f": 2}),
        # Very deep: ((a + b) * (c + d)) + ((e - f) * (g + h))
        (
            "((a + b) * (c + d)) + ((e - f) * (g + h))",
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 8, "f": 3, "g": 2, "h": 1},
        ),
    ]

    for expr_str, var_values in test_cases:
        print(f"\nTesting: {expr_str}")
        print(f"Variables: {var_values}")

        # Parse sympy expression
        expr = sympy.sympify(expr_str)

        # Calculate adequate dag_depth: need enough initial slots for all symbols
        num_symbols = len(expr.free_symbols)
        dag_depth = max(8, 2 * num_symbols + 4)  # +4 for operations buffer
        print(f"Symbols: {num_symbols}, using dag_depth: {dag_depth}")

        # Evaluate with sympy
        sympy_result = float(expr.subs(var_values))
        print(f"Sympy result: {sympy_result}")

        # Convert to DAG tensors
        V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

        # Set variable values in the tensors
        symbols = list(expr.free_symbols)
        for i, symbol in enumerate(sorted(symbols, key=str)):
            if i < V_mag.shape[2] and str(symbol) in var_values:
                V_mag[0, 0, i] = abs(var_values[str(symbol)])
                V_sign[0, 0, i] = 1.0 if var_values[str(symbol)] >= 0 else -1.0

        # Execute DAG
        executor = NewDAGExecutor(dag_depth=dag_depth)
        dag_result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
        dag_value = dag_result[0, 0].item()

        print(f"DAG result: {dag_value}")

        # Check if results match (within tolerance)
        tolerance = 1e-3
        if abs(sympy_result - dag_value) < tolerance:
            print("✅ PASS - Results match!")
        else:
            print(f"❌ FAIL - Results differ by {abs(sympy_result - dag_value)}")

        print(
            f"Relative error: {abs(sympy_result - dag_value) / max(abs(sympy_result), 1e-8):.6f}"
        )


def test_gradient_flow():
    """Test that gradients flow through the DAG execution process."""
    print("\n=== Testing Gradient Flow ===")

    # Create a simple expression: a * b + c
    a, b, c = sympy.symbols("a b c")
    expr = a * b + c

    # Convert to tensors
    V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=6)

    # Set up values and enable gradients
    V_mag = V_mag.clone().detach().requires_grad_(True)
    V_sign = V_sign.clone().detach().requires_grad_(True)
    O = O.clone().detach().requires_grad_(True)
    G = G.clone().detach().requires_grad_(True)

    # Set specific values: a=2, b=3, c=1
    with torch.no_grad():
        V_mag[0, 0, 0] = 2.0  # a
        V_mag[0, 0, 1] = 3.0  # b
        V_mag[0, 0, 2] = 1.0  # c

    print("Testing gradient flow through DAG execution...")

    # Execute DAG
    executor = NewDAGExecutor(dag_depth=6)
    result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

    # Compute a simple loss (difference from target)
    target = torch.tensor([[8.0]])  # Expected: 2*3 + 1 = 7, but we'll use 8 as target
    loss = torch.nn.MSELoss()(result, target)

    print(f"Result: {result[0, 0].item():.3f}")
    print(f"Target: {target[0, 0].item():.3f}")
    print(f"Loss: {loss.item():.6f}")

    # Backpropagate
    loss.backward()

    # Check gradients
    print("\nGradient Flow Analysis:")

    if V_mag.grad is not None:
        grad_norm = torch.norm(V_mag.grad).item()
        print(f"✅ V_mag gradients: norm = {grad_norm:.6f}")
        print(f"   V_mag.grad[0,0,:3] = {V_mag.grad[0, 0, :3].tolist()}")
    else:
        print("❌ No gradients for V_mag")

    if O.grad is not None:
        grad_norm = torch.norm(O.grad).item()
        print(f"✅ O gradients: norm = {grad_norm:.6f}")
        non_zero_grads = torch.sum(torch.abs(O.grad) > 1e-8).item()
        print(f"   Non-zero gradient elements in O: {non_zero_grads}")
    else:
        print("❌ No gradients for O")

    if G.grad is not None:
        grad_norm = torch.norm(G.grad).item()
        print(f"✅ G gradients: norm = {grad_norm:.6f}")
        print(f"   G.grad[0,0,:] = {G.grad[0, 0, :].tolist()}")
    else:
        print("❌ No gradients for G")

    # Test with predictor model gradients
    print("\nTesting predictor model gradient flow...")

    predictor = NewDAGPlanPredictor(dag_depth=6, hidden_dim=64)
    hidden = torch.randn(1, 1, 64, requires_grad=True)

    # Forward pass through predictor
    V_mag_pred, V_sign_pred, O_pred, G_pred = predictor(hidden)

    # Set values manually (as would happen in training)
    with torch.no_grad():
        V_mag_pred[0, 0, 0] = 2.0
        V_mag_pred[0, 0, 1] = 3.0
        V_mag_pred[0, 0, 2] = 1.0

    # Execute
    executor = NewDAGExecutor(dag_depth=6)
    result = executor.execute_with_plan(
        V_mag_pred, V_sign_pred, O_pred, G_pred, debug=False
    )

    # Loss and backprop
    loss = torch.nn.MSELoss()(result, target)
    loss.backward()

    # Check if gradients reached the predictor parameters
    total_grad_norm = 0
    param_count = 0
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm
            param_count += 1
            print(f"✅ {name}: grad norm = {grad_norm:.6f}")
        else:
            print(f"❌ {name}: no gradients")

    print(f"\nSummary: {param_count} parameters have gradients")
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    if total_grad_norm > 1e-6:
        print("🎉 SUCCESS: Gradients flow through the entire DAG pipeline!")
    else:
        print("⚠️  WARNING: Very small gradients detected")


if __name__ == "__main__":
    print("Testing New DAG Implementation with Domain Mixing")
    print("=" * 50)
    print()

    test_simple_arithmetic()
    test_subtraction_division()
    test_complex_expression()

    # Test expression_to_tensors conversion
    print("=== Testing Expression to Tensors Conversion ===")

    # Test simple expression: a + b
    a, b = sympy.symbols("a b")
    expr = a + b
    print(f"Converting expression: {expr}")

    V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=4)
    print(f"V_mag: {V_mag[0, 0]}")
    print(f"V_sign: {V_sign[0, 0]}")
    print(f"O[0]: {O[0, 0, 0]}")  # First operation
    print(f"G[0]: {G[0, 0, 0]}")  # Should be 1.0 for addition

    # Test by executing with real values
    executor = NewDAGExecutor(dag_depth=4)
    # Set a=2, b=3
    V_mag[0, 0, 0] = 2.0  # a
    V_mag[0, 0, 1] = 3.0  # b

    result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=True)
    print(f"Result: {result[0, 0].item():.3f} (expected = 5.0)")
    print()

    # Test multiplication: a * b
    expr = a * b
    print(f"Converting expression: {expr}")

    V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=4)
    print(f"O[0]: {O[0, 0, 0]}")  # First operation
    print(f"G[0]: {G[0, 0, 0]}")  # Should be 0.0 for multiplication

    # Test by executing
    V_mag[0, 0, 0] = 2.0  # a
    V_mag[0, 0, 1] = 3.0  # b

    result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=True)
    print(f"Result: {result[0, 0].item():.3f} (expected = 6.0)")
    print()

    # Test complex expressions with high depth
    test_complex_expressions_validation()

    # Test gradient flow
    test_gradient_flow()

    print("Testing complete!")
