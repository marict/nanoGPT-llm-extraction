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

        # Only set up hooks if model has done a forward pass
        if not hasattr(model, "last_activations"):
            return

        # Register hooks for standard activations
        for name, tensor in model.last_activations.items():
            if tensor is not None and tensor.requires_grad:

                def make_hook(n=name):
                    def save_grad(grad):
                        self.captured_gradients[n] = grad.detach().mean().item()

                    return save_grad

                hook = tensor.register_hook(make_hook())
                self.gradient_hooks.append(hook)

        # Special handling for op_logits gradients
        if (
            hasattr(model.dag.controller, "last_op_logits_with_grad")
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

        # Add entropy metrics
        if hasattr(model, "last_activations"):
            for name, act in model.last_activations.items():
                if act is not None:
                    with torch.no_grad():
                        prob = torch.softmax(act, dim=-1)
                        ent = -(prob * (prob + 1e-8).log()).sum(-1).mean()
                        metrics[f"dag_entropy/{name}"] = ent.item()

        # Add gradient metrics
        for grad_name, grad_val in self.captured_gradients.items():
            if grad_name.startswith("op_grad/") or grad_name == "op_logits_mean":
                metrics[grad_name] = grad_val
            else:
                metrics[f"dag_grad/{grad_name}"] = grad_val

        return metrics

    def get_op_probabilities(self, model) -> Dict[str, float]:
        """
        Get operation probabilities from the model.

        Args:
            model: DAGGPT model instance

        Returns:
            Dictionary of operation probabilities
        """
        if (
            not hasattr(model, "last_activations")
            or "op_logits" not in model.last_activations
        ):
            return {}

        op_logits = model.last_activations["op_logits"]
        if op_logits is None:
            return {}

        with torch.no_grad():
            # Take the first sample in the batch and convert to probabilities
            probs = F.softmax(op_logits, dim=-1)[0].detach().cpu().numpy()
            return {f"op_probs/{op}": float(p) for op, p in zip(op_names, probs)}

    def get_operand_probabilities(self, model) -> Dict[str, float]:
        """
        Get operand selection probabilities from the model.

        Args:
            model: DAGGPT model instance

        Returns:
            Dictionary of operand probabilities
        """
        if (
            not hasattr(model, "dag")
            or not hasattr(model.dag, "controller")
            or not hasattr(model.dag.controller, "last_attn")
        ):
            return {}

        last_attn = model.dag.controller.last_attn
        if last_attn is None:
            return {}

        with torch.no_grad():
            # last_attn is (B, 2, N) - batch, 2 operands, N nodes
            # Take the first sample in the batch
            attn_probs = last_attn[0].detach().cpu().numpy()  # (2, N)

            operand_probs = {}
            for operand_idx in range(attn_probs.shape[0]):  # 2 operands
                for node_idx in range(attn_probs.shape[1]):  # N nodes
                    prob = float(attn_probs[operand_idx, node_idx])
                    operand_probs[f"operand{operand_idx+1}_probs/node_{node_idx}"] = (
                        prob
                    )

            return operand_probs

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
        # Operation probabilities
        op_probs = self.get_op_probabilities(model)
        if op_probs:
            print("Operation probabilities:")
            for op_name, prob in op_probs.items():
                print(f"  {op_name.replace('op_probs/', '')}: {prob:.4f}")

        # Operand choices (instead of all probabilities)
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "controller")
            and hasattr(model.dag.controller, "last_attn")
            and model.dag.controller.last_attn is not None
        ):
            with torch.no_grad():
                # last_attn is (B, 2, N) - batch, 2 operands, N nodes
                # Take the first sample in the batch
                attn_probs = (
                    model.dag.controller.last_attn[0].detach().cpu().numpy()
                )  # (2, N)

                # Find the chosen operands (highest probability for each)
                operand1_choice = int(attn_probs[0].argmax())
                operand2_choice = int(attn_probs[1].argmax())
                max_nodes = attn_probs.shape[1]

                print(
                    f"Operands chosen: {operand1_choice}, {operand2_choice} (out of {max_nodes})"
                )

        # Node values
        node_values = self.get_node_values_list(model)
        if node_values:
            print(f"Node values: {[f'{val:.4f}' for val in node_values]}")

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
        log_dict.update(self.get_op_probabilities(model))
        log_dict.update(self.get_operand_probabilities(model))

        return log_dict
