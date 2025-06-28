"""
DAG Logging Module

This module contains all logging-related functionality for DAG models,
separated from the core training and inference code.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

import wandb
from dag_model import op_names


class DAGLogger:
    """
    Handles all logging functionality for DAG models.

    This class extracts logging information from DAG models without
    polluting the core model code with logging-specific logic.
    """

    def __init__(self):
        self.gradient_hooks = []
        self.captured_gradients = {}

    def setup_gradient_tracking(self, model) -> None:
        """
        Set up gradient tracking hooks for a DAG model.

        Args:
            model: DAGGPT model instance
        """
        self.clear_gradient_hooks()
        self.captured_gradients = {}

        # For operation gradients, we'll set up a hook to capture them
        # during the forward pass when they're actually used
        self._setup_op_grad_tracking(model)

        # Set up DAG hidden gradient tracking (works if called after forward pass)
        if (
            hasattr(model, "last_dag_hidden")
            and model.last_dag_hidden is not None
            and model.last_dag_hidden.requires_grad
        ):

            def save_dag_hidden_grad(grad):
                grad_norm = grad.detach().norm().item()
                grad_mean = grad.detach().mean().item()
                grad_std = grad.detach().std().item()
                grad_max = grad.detach().max().item()
                grad_min = grad.detach().min().item()

                self.captured_gradients["dag_hidden_grad_norm"] = grad_norm
                self.captured_gradients["dag_hidden_grad_mean"] = grad_mean
                self.captured_gradients["dag_hidden_grad_std"] = grad_std
                self.captured_gradients["dag_hidden_grad_max"] = grad_max
                self.captured_gradients["dag_hidden_grad_min"] = grad_min

            hook = model.last_dag_hidden.register_hook(save_dag_hidden_grad)
            self.gradient_hooks.append(hook)

    def update_gradient_tracking(self, model):
        """Update gradient tracking after forward pass when tensors are available."""
        # Set up operation gradient tracking now that tensors exist
        self._setup_op_grad_tracking(model)

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

        for grad_name, grad_val in self.captured_gradients.items():
            if grad_name.startswith("op_grad/") or grad_name == "op_logits_mean":
                metrics[grad_name] = grad_val
            else:
                metrics[f"dag_grad/{grad_name}"] = grad_val

        if hasattr(model, "last_gate_values") and model.last_gate_values is not None:
            gate_values = model.last_gate_values
            metrics["gate/mean"] = gate_values.mean().item()
            metrics["gate/min"] = gate_values.min().item()
            metrics["gate/max"] = gate_values.max().item()
            close_to_zero = (gate_values < 0.1).float().mean().item()
            metrics["gate/close_to_zero_ratio"] = close_to_zero

        if hasattr(model, "last_norm_values") and model.last_norm_values is not None:
            for norm_name, norm_val in model.last_norm_values.items():
                metrics[f"norm/{norm_name}"] = norm_val.item()

        return metrics

    def get_node_values_list(self, model) -> List[float]:
        """
        Get node values from the model as a list.

        Args:
            model: GPT model instance

        Returns:
            List of node values
        """
        return model.get_node_values_list()

    def get_detailed_node_values(self, model) -> dict:
        """
        Get detailed node values from the model showing all scratch nodes per token.

        Args:
            model: GPT model instance

        Returns:
            Dictionary with detailed node information
        """
        if hasattr(model, "get_detailed_node_values"):
            return model.get_detailed_node_values()
        return {}

    def format_console_logging(self, model) -> None:
        """
        Print formatted logging information to console.

        Args:
            model: DAGGPT model instance
        """
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "plan_predictor")
            and hasattr(model.dag.plan_predictor, "last_op_logits")
            and model.dag.plan_predictor.last_op_logits is not None
        ):
            with torch.no_grad():
                op_logits = model.dag.plan_predictor.last_op_logits
                probs = F.softmax(op_logits, dim=-1).detach().cpu().numpy()

                print("Operation probabilities:")
                for op_name, prob in zip(op_names, probs):
                    print(f"  {op_name}: {prob:.4f}")

        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "plan_predictor")
            and hasattr(model.dag.plan_predictor, "last_attn")
            and model.dag.plan_predictor.last_attn is not None
        ):
            with torch.no_grad():
                attn_probs = model.dag.plan_predictor.last_attn[0]

                print("Operands chosen:")
                # attn_probs shape: (T, dag_depth, max_nodes, 2)
                # where the last dimension is [operand1_probs, operand2_probs]
                if attn_probs.dim() >= 3 and attn_probs.shape[-1] == 2:
                    for t in range(attn_probs.shape[0]):  # time steps
                        for step in range(attn_probs.shape[1]):  # dag steps
                            operand1_probs = attn_probs[
                                t, step, :, 0
                            ]  # operand1 probabilities over nodes
                            operand2_probs = attn_probs[
                                t, step, :, 1
                            ]  # operand2 probabilities over nodes
                            operand1_choice = int(operand1_probs.argmax())
                            operand2_choice = int(operand2_probs.argmax())
                            max_nodes = operand1_probs.shape[0]
                            print(
                                f"  Token {t}, Step {step}: {operand1_choice}, {operand2_choice} (out of {max_nodes})"
                            )
                else:
                    print(
                        "  [Attention probabilities format not supported for display]"
                    )

        # Show detailed node values if available
        detailed_values = self.get_detailed_node_values(model)
        if detailed_values and detailed_values.get("values_per_token"):
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
        else:
            # Fallback to simple values
            node_values = self.get_node_values_list(model)
            if node_values:
                print("Node values per token position:")
                for t, val in enumerate(node_values):
                    print(f"  Token {t}: {val:.4f}")

        if hasattr(model, "last_gate_values") and model.last_gate_values is not None:
            gate_values = model.last_gate_values
            gate_mean = gate_values.mean().item()
            close_to_zero = (gate_values < 0.1).float().mean().item()
            print(
                f"Gate values: mean={gate_mean:.4f}, close_to_zero_ratio={close_to_zero:.4f}"
            )

        # Norm values
        if hasattr(model, "last_norm_values") and model.last_norm_values is not None:
            norm_strs = [
                f"{name}={val.item():.4f}"
                for name, val in model.last_norm_values.items()
            ]
            print(f"Norms: {', '.join(norm_strs)}")

        # DAG hidden gradients (if available)
        dag_grad_keys = [
            k for k in self.captured_gradients.keys() if k.startswith("dag_hidden_grad")
        ]
        if dag_grad_keys:
            grad_strs = [
                f"{k.replace('dag_hidden_grad_', '')}={self.captured_gradients[k]:.6f}"
                for k in dag_grad_keys
            ]
            print(f"DAG hidden gradients: {', '.join(grad_strs)}")

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
        """Set up operation gradient tracking using the saved tensor from forward pass."""
        # Try to hook into the full operation probabilities tensor which should be connected to the computation graph
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "plan_predictor")
            and hasattr(model.dag.plan_predictor, "last_operation_probs_full")
            and model.dag.plan_predictor.last_operation_probs_full is not None
            and model.dag.plan_predictor.last_operation_probs_full.requires_grad
        ):

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

        # Fallback to the mean version if full version not available
        elif (
            hasattr(model, "dag")
            and hasattr(model.dag, "plan_predictor")
            and hasattr(model.dag.plan_predictor, "last_op_logits_with_grad")
            and model.dag.plan_predictor.last_op_logits_with_grad is not None
            and model.dag.plan_predictor.last_op_logits_with_grad.requires_grad
        ):

            def save_op_grad(grad):
                self.captured_gradients["op_logits_mean"] = grad.detach().mean().item()
                op_grads = grad.detach().cpu().numpy()
                for i, op_name in enumerate(op_names):
                    self.captured_gradients[f"op_grad/{op_name}"] = float(op_grads[i])

            hook = model.dag.plan_predictor.last_op_logits_with_grad.register_hook(
                save_op_grad
            )
            self.gradient_hooks.append(hook)
