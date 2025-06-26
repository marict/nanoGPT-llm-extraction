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
        # Clear any existing hooks
        self.clear_gradient_hooks()
        self.captured_gradients = {}

        # Special handling for op_logits gradients
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "controller")
            and hasattr(model.dag.controller, "last_op_logits_with_grad")
            and model.dag.controller.last_op_logits_with_grad is not None
            and model.dag.controller.last_op_logits_with_grad.requires_grad
        ):

            def save_op_grad(grad):
                # Save both mean and per-operation gradients
                self.captured_gradients["op_logits_mean"] = grad.detach().mean().item()
                # Save individual operation gradients
                op_grads = grad.detach().mean(dim=0).cpu().numpy()  # Average over batch
                for i, op_name in enumerate(op_names):
                    self.captured_gradients[f"op_grad/{op_name}"] = float(op_grads[i])

            hook = model.dag.controller.last_op_logits_with_grad.register_hook(
                save_op_grad
            )
            self.gradient_hooks.append(hook)

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

        # Add gradient metrics
        for grad_name, grad_val in self.captured_gradients.items():
            if grad_name.startswith("op_grad/") or grad_name == "op_logits_mean":
                metrics[grad_name] = grad_val
            else:
                metrics[f"dag_grad/{grad_name}"] = grad_val

        # Add gate values (if available)
        if hasattr(model, "last_gate_values") and model.last_gate_values is not None:
            gate_values = model.last_gate_values
            metrics["gate/mean"] = gate_values.mean().item()
            metrics["gate/min"] = gate_values.min().item()
            metrics["gate/max"] = gate_values.max().item()
            # Check if gate is close to 0 (DAG not contributing)
            close_to_zero = (gate_values < 0.1).float().mean().item()
            metrics["gate/close_to_zero_ratio"] = close_to_zero

        # Add norm values (if available)
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
        # Delegate to the model's own implementation to avoid duplication
        return model.get_node_values_list()

    def format_console_logging(self, model) -> None:
        """
        Print formatted logging information to console.

        Args:
            model: DAGGPT model instance
        """
        # Operation probabilities - calculate directly for console output
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "controller")
            and hasattr(model.dag.controller, "last_op_logits")
            and model.dag.controller.last_op_logits is not None
        ):
            with torch.no_grad():
                op_logits = model.dag.controller.last_op_logits
                # Take the first sample in the batch and convert to probabilities
                probs = F.softmax(op_logits, dim=-1)[0].detach().cpu().numpy()

                print("Operation probabilities:")
                for op_name, prob in zip(op_names, probs):
                    print(f"  {op_name}: {prob:.4f}")

        # Operand choices (instead of all probabilities)
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "controller")
            and hasattr(model.dag.controller, "last_attn")
            and model.dag.controller.last_attn is not None
        ):
            with torch.no_grad():
                # last_attn is (B, T, 2, N) - batch, time, 2 operands, N nodes
                # Take the first sample in the batch
                attn_probs = model.dag.controller.last_attn[0]  # (T, 2, N)

                print("Operands chosen per token position:")
                for t in range(attn_probs.shape[0]):
                    # Find the chosen operands for this token
                    operand1_choice = int(attn_probs[t, 0].argmax())
                    operand2_choice = int(attn_probs[t, 1].argmax())
                    max_nodes = attn_probs.shape[-1]
                    print(
                        f"  Token {t}: {operand1_choice}, {operand2_choice} (out of {max_nodes})"
                    )

        # Node values (now per token)
        node_values = self.get_node_values_list(model)
        if node_values:
            print("Node values per token position:")
            for t, val in enumerate(node_values):
                print(f"  Token {t}: {val:.4f}")

        # Gate values
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
