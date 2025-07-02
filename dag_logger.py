"""
DAG Logging Module

This module contains all logging-related functionality for DAG models,
separated from the core training and inference code.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from dag_model import op_names


class DAGLogger:
    """
    Handles all logging functionality for DAG models.

    This class extracts logging information from DAG models without
    polluting the core model code with logging-specific logic.

    NOTE: When making additions/modifications to logging functionality, logging comes first. If a state is invalid for logging, we should stop training and throw an error.
    """

    def __init__(self):
        self.gradient_hooks = []
        self.captured_gradients = {}
        self.logging_data = {}  # Single dictionary for all logging data

    def setup_gradient_tracking(self, model) -> None:
        """
        Set up gradient tracking hooks for a DAG model.

        Args:
            model: DAGGPT model instance

        Raises:
            AssertionError: If required tensors for gradient tracking are not available
        """
        self.clear_gradient_hooks()
        self.captured_gradients = {}

        # For operation gradients, we'll set up a hook to capture them
        # during the forward pass when they're actually used
        self._setup_op_grad_tracking(model)

        # Assert DAG output gradient tracking requirements - use DAG attributes directly
        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(model.dag, "final_hidden"), "DAG missing final_hidden attribute"
        assert model.dag.final_hidden is not None, "dag.final_hidden is None"
        assert (
            model.dag.final_hidden.requires_grad
        ), "dag.final_hidden does not require gradients"

        def save_dag_output_grad(grad):
            grad_norm = grad.detach().norm().item()
            grad_mean = grad.detach().mean().item()
            grad_std = grad.detach().std().item()

            self.captured_gradients["dag_output_grad_norm"] = grad_norm
            self.captured_gradients["dag_output_grad_mean"] = grad_mean
            self.captured_gradients["dag_output_grad_std"] = grad_std

        hook = model.dag.final_hidden.register_hook(save_dag_output_grad)
        self.gradient_hooks.append(hook)

        # Assert DAG scratch values gradient tracking requirements - use DAG attributes directly
        assert hasattr(model.dag, "final_values"), "DAG missing final_values attribute"
        assert model.dag.final_values is not None, "dag.final_values is None"
        assert (
            model.dag.final_values.requires_grad
        ), "dag.final_values does not require gradients"

        def save_dag_scratch_grad(grad):
            grad_norm = grad.detach().norm().item()
            grad_mean = grad.detach().mean().item()

            self.captured_gradients["dag_scratch_grad_norm"] = grad_norm
            self.captured_gradients["dag_scratch_grad_mean"] = grad_mean

        hook = model.dag.final_values.register_hook(save_dag_scratch_grad)
        self.gradient_hooks.append(hook)

        # Capture gradients of original (pre-DAG) hidden states (mandatory)
        if (
            not hasattr(model, "last_original_hidden")
            or model.last_original_hidden is None
        ):
            raise RuntimeError(
                "Model missing 'last_original_hidden'; cannot log. Halting training."
            )

        assert (
            model.last_original_hidden.requires_grad
        ), "last_original_hidden does not require gradients"

        def save_orig_hidden_grad(grad):
            self.captured_gradients["orig_hidden_grad_mean"] = (
                grad.detach().mean().item()
            )

        hook = model.last_original_hidden.register_hook(save_orig_hidden_grad)
        self.gradient_hooks.append(hook)

    def update_gradient_tracking(self, model):
        """Update gradient tracking after forward pass when tensors are available."""
        # Set up operation gradient tracking now that tensors exist
        self._setup_op_grad_tracking(model)

    def compute_log_statistics(self, model):
        """
        Compute and store ALL logging statistics from model forward pass into a single dictionary.
        Extracts values directly from the model's stored state.

        Args:
            model: The GPT model that has completed a forward pass
        """
        # Clear previous logging data
        self.logging_data = {}

        if model.config.dag_depth == 0:
            return

        # Logging is mandatory: ensure plan_predictor and probability tensor are available
        if not hasattr(model.dag, "plan_predictor"):
            raise RuntimeError("DAG model missing 'plan_predictor'; cannot log.")

        if not hasattr(model.dag.plan_predictor, "last_operation_probs_full"):
            raise RuntimeError(
                "plan_predictor missing 'last_operation_probs_full'; cannot log gradients/probs."
            )

        if model.dag.plan_predictor.last_operation_probs_full is None:
            raise RuntimeError(
                "'last_operation_probs_full' is None – logging disabled. Aborting to avoid waste."
            )

        # Extract node values (both traditional and detailed)
        self.logging_data["node_values"] = self.get_node_values_list(model)
        self.logging_data["detailed_node_values"] = self.get_detailed_node_values(model)

        # Extract norm values from the model's stored hidden states
        original_hidden = model.last_original_hidden
        dag_hidden = model.dag.final_hidden
        mixed_hidden = model.last_mixed_hidden
        if (
            original_hidden is not None
            and dag_hidden is not None
            and mixed_hidden is not None
        ):
            norm_values = {
                "hidden": original_hidden[:, -1].norm(dim=-1).mean().detach().item(),
                "dag_hidden": dag_hidden[:, -1].norm(dim=-1).mean().detach().item(),
                "fused": mixed_hidden[:, -1].norm(dim=-1).mean().detach().item(),
            }
            self.logging_data["norm_values"] = norm_values

        # Operation probability statistics (average over batch, time and DAG steps)
        with torch.no_grad():
            probs = model.dag.plan_predictor.last_operation_probs_full.detach()
            mean_probs = probs.mean(dim=(0, 1, 2))  # (n_ops,)
            self.logging_data["op_probs"] = {
                op_name: float(mean_probs[i].item())
                for i, op_name in enumerate(op_names)
            }

    def clear_gradient_hooks(self) -> None:
        """Remove all registered gradient hooks."""
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()

    def get_extra_vals(self, model) -> Dict[str, float]:
        """
        Get all extra logging values for a model (entropies + gradients).

        Args:
            model: GPT model instance

        Returns:
            Dictionary of all logging metrics
        """
        metrics = {}

        # Add gradient information - preserve original names
        for grad_name, grad_val in self.captured_gradients.items():
            metrics[grad_name] = grad_val

        # Use centralized logging data - no fallbacks
        assert (
            self.logging_data
        ), "Logging data not available - call compute_log_statistics() first"

        if "norm_values" in self.logging_data:
            norm_data = self.logging_data["norm_values"]
            for norm_name, norm_val in norm_data.items():
                metrics[f"norm/{norm_name}"] = norm_val

        if "op_probs" in self.logging_data:
            for op_name, p_val in self.logging_data["op_probs"].items():
                metrics[f"op_prob/{op_name}"] = p_val

        return metrics

    def get_node_values_list(self, model) -> List[float]:
        """
        Get node values from the model as a list.

        Args:
            model: GPT model instance

        Returns:
            List of node values
        """
        if model.config.dag_depth == 0:
            return []

        assert hasattr(
            model, "last_values_list"
        ), "Model missing last_values_list attribute"
        assert model.last_values_list is not None, "last_values_list is None"

        # Extract values from the stored tensor
        assert isinstance(
            model.last_values_list, torch.Tensor
        ), "last_values_list must be a tensor"

        if model.last_values_list.dim() == 3:  # (B, scratch_nodes, T)
            B, scratch_nodes, T = model.last_values_list.shape
            # Get the final slot for each token (most recent computation)
            final_slot = (
                (model.config.dag_depth - 1) % scratch_nodes
                if model.config.dag_depth > 0
                else 1
            )
            # Extract values from the final slot and average across batch
            final_values = model.last_values_list[:, final_slot, :]  # (B, T)
            return [val.mean().item() for val in final_values.transpose(0, 1)]
        else:  # (B, T) format
            return [val.mean().item() for val in model.last_values_list.transpose(0, 1)]

    def get_detailed_node_values(self, model) -> dict:
        """
        Get detailed node values from the model showing all scratch nodes per token.

        Args:
            model: GPT model instance

        Returns:
            Dictionary with detailed node information, empty dict if no forward pass or dag_depth=0.
        """
        if model.config.dag_depth == 0:
            return {}

        assert hasattr(
            model, "last_values_list"
        ), "Model missing last_values_list attribute"
        assert model.last_values_list is not None, "last_values_list is None"
        assert isinstance(
            model.last_values_list, torch.Tensor
        ), "last_values_list must be a tensor"
        assert (
            model.last_values_list.dim() == 3
        ), f"Expected 3D tensor, got {model.last_values_list.dim()}D"

        B, scratch_nodes, T = model.last_values_list.shape

        # Get values for all scratch nodes per token, averaged across batch
        values_per_token = []
        aggregated_per_token = []

        for t in range(T):
            token_values = []
            for s in range(scratch_nodes):
                # Average across batch dimension
                value = model.last_values_list[:, s, t].mean().item()
                token_values.append(value)

            values_per_token.append(token_values)
            # Aggregate all scratch nodes for this token (mean)
            aggregated_per_token.append(sum(token_values) / len(token_values))

        return {
            "values_per_token": values_per_token,
            "aggregated_per_token": aggregated_per_token,
            "scratch_nodes": scratch_nodes,
            "sequence_length": T,
            "batch_size": B,
        }

    def format_console_logging(self, model) -> None:
        """
        Print formatted logging information to console.

        Args:
            model: DAGGPT model instance
        """
        # Assert required model structure
        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(
            model.dag, "plan_predictor"
        ), "DAG missing plan_predictor attribute"

        if (
            hasattr(model.dag.plan_predictor, "last_op_logits")
            and model.dag.plan_predictor.last_op_logits is not None
        ):
            with torch.no_grad():
                op_logits = model.dag.plan_predictor.last_op_logits
                probs = F.softmax(op_logits, dim=-1).detach().cpu().numpy()

                print("Operation probabilities:")
                for op_name, prob in zip(op_names, probs):
                    print(f"  {op_name}: {prob:.4f}")

        if (
            hasattr(model.dag.plan_predictor, "last_attn")
            and model.dag.plan_predictor.last_attn is not None
        ):
            with torch.no_grad():
                attn_probs = model.dag.plan_predictor.last_attn[0]

                print("Operands chosen:")
                assert (
                    attn_probs.dim() >= 3 and attn_probs.shape[-1] == 2
                ), "Unexpected attention probabilities format"

                for t in range(attn_probs.shape[0]):  # time steps
                    for step in range(attn_probs.shape[1]):  # dag steps
                        operand1_probs = attn_probs[t, step, :, 0]
                        operand2_probs = attn_probs[t, step, :, 1]
                        operand1_choice = int(operand1_probs.argmax())
                        operand2_choice = int(operand2_probs.argmax())
                        max_nodes = model.dag.scratch_nodes * (t + 1)
                        print(
                            f"  Token {t}, Step {step}: {operand1_choice}, {operand2_choice} (out of {max_nodes})"
                        )

        # Show detailed node values - no fallbacks
        assert (
            self.logging_data
        ), "Logging data not available - call compute_log_statistics() first"
        assert (
            "detailed_node_values" in self.logging_data
        ), "Detailed node values not in logging data"

        detailed_values = self.logging_data["detailed_node_values"]
        assert detailed_values and detailed_values.get(
            "values_per_token"
        ), "Invalid detailed node values"

        print("Node values per token position:")
        scratch_nodes = detailed_values["scratch_nodes"]
        for t, token_values in enumerate(detailed_values["values_per_token"]):
            if scratch_nodes > 1:
                values_str = ", ".join(
                    [f"slot{s}: {val:.4f}" for s, val in enumerate(token_values)]
                )
                aggregated = detailed_values["aggregated_per_token"][t]
                print(f"  Token {t}: [{values_str}] (avg: {aggregated:.4f})")
            else:
                print(f"  Token {t}: {token_values[0]:.4f}")

        # Norm values - no fallbacks
        assert "norm_values" in self.logging_data, "Norm values not in logging data"
        norm_data = self.logging_data["norm_values"]
        norm_strs = [f"{name}={val:.4f}" for name, val in norm_data.items()]
        print(f"Norms: {', '.join(norm_strs)}")

        # DAG gradients
        dag_grad_keys = [
            k
            for k in self.captured_gradients.keys()
            if k.startswith("dag_output_grad") or k.startswith("dag_scratch_grad")
        ]
        if dag_grad_keys:
            print("DAG gradients:")
            for k in dag_grad_keys:
                print(f"  {k}: {self.captured_gradients[k]:.6f}")

    def get_wandb_logging_dict(self, model, base_dict: Optional[Dict] = None) -> Dict:
        """
        Get dictionary suitable for wandb logging.

        Args:
            model: DAGGPT model instance
            base_dict: Optional base dictionary to extend

        Returns:
            Dictionary ready for wandb logging
        """
        if base_dict is None:
            base_dict = {}

        # Add all logging metrics
        log_dict = dict(base_dict)
        log_dict.update(self.get_extra_vals(model))

        return log_dict

    def _setup_op_grad_tracking(self, model):
        """
        Set up operation gradient tracking using the saved tensor from forward pass.

        Raises:
            AssertionError: If required operation probability tensors are not available
        """
        # Assert all required conditions - no fallbacks
        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(
            model.dag, "plan_predictor"
        ), "DAG missing plan_predictor attribute"
        assert hasattr(
            model.dag.plan_predictor, "last_operation_probs_full"
        ), "Plan predictor missing last_operation_probs_full"
        assert (
            model.dag.plan_predictor.last_operation_probs_full is not None
        ), "last_operation_probs_full is None"
        assert (
            model.dag.plan_predictor.last_operation_probs_full.requires_grad
        ), "last_operation_probs_full does not require gradients"

        def save_op_grad(grad):
            # grad shape: (B, T, dag_depth, n_ops)
            self.captured_gradients["op_logits_mean"] = grad.detach().mean().item()
            # Average over batch, time, and steps to get per-operation gradients
            op_grads = grad.detach().mean(dim=(0, 1, 2)).cpu().numpy()  # (n_ops,)
            for i, op_name in enumerate(op_names):
                self.captured_gradients[f"op_grad/{op_name}"] = float(op_grads[i])

        hook = model.dag.plan_predictor.last_operation_probs_full.register_hook(
            save_op_grad
        )
        self.gradient_hooks.append(hook)
