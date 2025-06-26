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

        # Time-series data for tracking probabilities over training
        self.step_counter = 0
        self.operation_prob_history = {op: [] for op in op_names}
        self.operand_prob_history = {}  # Will be initialized based on number of nodes
        self.step_history = []

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

    def update_probability_history(self, model) -> None:
        """
        Update the time-series history with current model probabilities.
        Call this during training to accumulate data for time-series plots.

        Args:
            model: DAGGPT model instance
        """
        # Update operation probabilities
        if (
            hasattr(model, "last_activations")
            and "op_logits" in model.last_activations
            and model.last_activations["op_logits"] is not None
        ):
            with torch.no_grad():
                op_logits = model.last_activations["op_logits"]
                probs = F.softmax(op_logits, dim=-1)[0].detach().cpu().numpy()

                for i, op_name in enumerate(op_names):
                    self.operation_prob_history[op_name].append(float(probs[i]))

        # Update operand probabilities
        if (
            hasattr(model, "dag")
            and hasattr(model.dag, "controller")
            and hasattr(model.dag.controller, "last_attn")
            and model.dag.controller.last_attn is not None
        ):
            with torch.no_grad():
                last_attn = model.dag.controller.last_attn
                attn_probs = last_attn[0].detach().cpu().numpy()  # (2, N)

                # Initialize operand history if not done yet
                if not self.operand_prob_history:
                    num_nodes = attn_probs.shape[1]
                    for operand_idx in range(2):  # 2 operands
                        for node_idx in range(num_nodes):
                            key = f"operand{operand_idx+1}_node{node_idx}"
                            self.operand_prob_history[key] = []

                # Update operand probabilities
                for operand_idx in range(attn_probs.shape[0]):  # 2 operands
                    for node_idx in range(attn_probs.shape[1]):  # N nodes
                        key = f"operand{operand_idx+1}_node{node_idx}"
                        prob = float(attn_probs[operand_idx, node_idx])
                        self.operand_prob_history[key].append(prob)

        # Update step counter and history
        self.step_history.append(self.step_counter)
        self.step_counter += 1

    def get_op_probabilities(self, model) -> Dict[str, object]:
        """
        Get operation probabilities as a time-series wandb plot showing evolution over training.

        Args:
            model: DAGGPT model instance

        Returns:
            Dictionary containing wandb time-series plot for operation probabilities
        """
        # Update the current step data first
        self.update_probability_history(model)

        # Create plot with whatever data we have (even single point)
        if len(self.step_history) == 0:
            return {}

        # Create time-series data for wandb
        # Organize data by operation for line_series format
        xs = self.step_history  # Single x array for all operations
        ys = []  # List of y arrays, one per operation
        keys = []  # Operation names

        for op_name in op_names:
            if op_name in self.operation_prob_history:
                history = self.operation_prob_history[op_name]
                # Only include data points we have
                if len(history) >= len(self.step_history):
                    ys.append(history[: len(self.step_history)])
                    keys.append(op_name)

        if not ys:
            return {}

        # Create a multi-line time series plot
        line_plot = wandb.plot.line_series(
            xs=xs,  # Single x array (training steps)
            ys=ys,  # List of y arrays (probabilities for each operation)
            keys=keys,  # Operation names for legend
            title="Operation Probabilities Over Training",
            xname="Training Step",
        )

        return {"op_probs_timeseries": line_plot}

    def get_operand_probabilities(self, model) -> Dict[str, object]:
        """
        Get operand selection probabilities as time-series wandb plots showing evolution over training.

        Args:
            model: DAGGPT model instance

        Returns:
            Dictionary containing wandb time-series plots for operand probabilities
        """
        # Update history if we don't have enough data points yet
        if len(self.step_history) == 0:
            self.update_probability_history(model)

        # Create plot with whatever data we have
        if len(self.step_history) == 0 or not self.operand_prob_history:
            return {}

        operand_plots = {}

        # Create separate time-series plots for each operand
        for operand_idx in range(2):  # 2 operands
            xs = self.step_history  # Single x array for all nodes
            ys = []  # List of y arrays, one per node
            keys = []  # Node names

            # Collect data for all nodes of this operand
            for key, history in self.operand_prob_history.items():
                if key.startswith(f"operand{operand_idx+1}_"):
                    node_name = key.replace(f"operand{operand_idx+1}_", "")
                    # Only include data points we have
                    if len(history) >= len(self.step_history):
                        ys.append(history[: len(self.step_history)])
                        keys.append(node_name)

            if ys:
                line_plot = wandb.plot.line_series(
                    xs=xs,  # Single x array (training steps)
                    ys=ys,  # List of y arrays (probabilities for each node)
                    keys=keys,  # Node names for legend
                    title=f"Operand {operand_idx + 1} Node Probabilities Over Training",
                    xname="Training Step",
                )

                operand_plots[f"operand{operand_idx+1}_probs_timeseries"] = line_plot

        return operand_plots

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
            hasattr(model, "last_activations")
            and "op_logits" in model.last_activations
            and model.last_activations["op_logits"] is not None
        ):
            with torch.no_grad():
                op_logits = model.last_activations["op_logits"]
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

    def log_training_step(self, model) -> None:
        """
        Log probabilities for a single training step.
        Call this method after each forward pass during training to accumulate
        time-series data for plotting.

        Args:
            model: DAGGPT model instance
        """
        self.update_probability_history(model)

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
