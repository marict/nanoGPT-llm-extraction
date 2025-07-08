"""
DAG Logging Module

This module contains all logging-related functionality for DAG models,
separated from the core training and inference code.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from models.dag_model import OP_NAMES


class DAGLogger:
    """
    Handles all logging functionality for DAG models.

    This class extracts logging information from DAG models without
    polluting the core model code with logging-specific logic.
    """

    def __init__(self):
        """Initialize DAG logger."""
        self.logging_data = {}
        self.captured_gradients = {}
        self.gradient_hooks = []
        self.op_grad_hook = None

    def setup_gradient_tracking(self, model) -> None:
        """Set up gradient tracking hooks for a DAG model."""
        if model.config.dag_depth == 0:
            return

        self.clear_gradient_hooks()
        self.captured_gradients = {}

        self._setup_op_grad_tracking(model)

        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(model.dag, "final_hidden"), "DAG missing final_hidden attribute"
        assert model.dag.final_hidden is not None, "dag.final_hidden is None"

        if model.dag.final_hidden.requires_grad:
            hook = model.dag.final_hidden.register_hook(self._dag_output_grad_hook_fn)
            self.gradient_hooks.append(hook)

        if hasattr(model.dag, "final_values") and model.dag.final_values is not None:
            if model.dag.final_values.requires_grad:
                hook = model.dag.final_values.register_hook(
                    self._dag_final_values_grad_hook_fn
                )
                self.gradient_hooks.append(hook)

        if (
            hasattr(model.dag, "original_hidden")
            and model.dag.original_hidden is not None
        ):
            if model.dag.original_hidden.requires_grad:
                hook = model.dag.original_hidden.register_hook(
                    self._orig_hidden_grad_hook_fn
                )
                self.gradient_hooks.append(hook)

        if hasattr(model.dag, "gate_w_d") and model.dag.gate_w_d is not None:
            if model.dag.gate_w_d.requires_grad:
                hook = model.dag.gate_w_d.register_hook(self._gate_w_d_grad_hook_fn)
                self.gradient_hooks.append(hook)

        if hasattr(model.dag, "gate_w_o") and model.dag.gate_w_o is not None:
            if model.dag.gate_w_o.requires_grad:
                hook = model.dag.gate_w_o.register_hook(self._gate_w_o_grad_hook_fn)
                self.gradient_hooks.append(hook)

    def update_gradient_tracking(self, model):
        """Update gradient tracking after forward pass when tensors are available."""
        self._setup_op_grad_tracking(model)

    def compute_log_statistics(self, model):
        """
        Compute and store all logging statistics from model forward pass.

        Args:
            model: The GPT model that has completed a forward pass
        """
        self.logging_data = {}
        if model.config.dag_depth == 0:
            return

        if not hasattr(model.dag, "plan_predictor"):
            raise RuntimeError("DAG model missing 'plan_predictor'")

        if not hasattr(model.dag.plan_predictor, "last_operation_probs"):
            raise RuntimeError("plan_predictor missing 'last_operation_probs'")

        if model.dag.plan_predictor.last_operation_probs is None:
            raise RuntimeError("'last_operation_probs' is None")

        self.logging_data["node_values"] = self.get_node_values_list(model)
        self.logging_data["detailed_node_values"] = self.get_detailed_node_values(model)

        original_hidden = model.last_original_hidden
        dag_hidden = model.dag.final_hidden
        mixed_hidden = model.last_mixed_hidden
        final_values = model.final_values
        last_gate = model.last_gate
        mag_logits = model.dag.plan_predictor.mag_logits

        if original_hidden is None:
            raise RuntimeError("model missing original_hidden")
        if dag_hidden is None:
            raise RuntimeError("dag missing final_hidden")
        if mixed_hidden is None:
            raise RuntimeError("model missing last_mixed_hidden")
        if final_values is None:
            raise RuntimeError("model missing final_values")
        if last_gate is None:
            raise RuntimeError("model missing last_gate")
        if mag_logits is None:
            raise RuntimeError("plan_predictor missing mag_logits")

        hidden_norm = original_hidden[:, -1].norm(dim=-1).mean().detach().item()
        dag_norm = dag_hidden[:, -1].norm(dim=-1).mean().detach().item()
        fused_norm = mixed_hidden[:, -1].norm(dim=-1).mean().detach().item()
        final_values_norm = final_values.norm(dim=-1).mean().detach().item()
        last_gate_norm = last_gate.norm(dim=-1).mean().detach().item()
        mag_logits_norm = mag_logits.norm(dim=-1).mean().detach().item()

        norm_values = {
            "original_hidden": hidden_norm,
            "dag_hidden": dag_norm,
            "fused": fused_norm,
            "dag_to_orig_ratio": dag_norm / (hidden_norm + 1e-8),
            "final_values": final_values_norm,
            "last_gate": last_gate_norm,
            "mag_logits": mag_logits_norm,
        }
        self.logging_data["norm_values"] = norm_values

        gate_mean = last_gate.mean().item()
        gate_max = last_gate.max().item()
        self.logging_data["gate_mean"] = gate_mean
        self.logging_data["gate_max"] = gate_max
        self.logging_data["last_gate"] = last_gate.detach().clone()

        with torch.no_grad():
            if model.dag.plan_predictor.last_operation_probs is None:
                raise RuntimeError("last_operation_probs is None")
            probs = model.dag.plan_predictor.last_operation_probs.detach()
            mean_probs = probs.mean(dim=(0, 1, 2))
            self.logging_data["op_probs"] = {
                op_name: float(mean_probs[i].item())
                for i, op_name in enumerate(OP_NAMES)
            }

        op_metrics = self._collect_op_metrics(model)
        self.logging_data["op_metrics"] = op_metrics

    def clear_gradient_hooks(self) -> None:
        """Remove all registered gradient hooks."""
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()

    def get_extra_vals(self, model) -> Dict[str, float]:
        """Get extra values for logging."""
        if model.config.dag_depth == 0:
            return {}

        op_metrics = self._collect_op_metrics(model)

        grad_metrics = {}
        if hasattr(self, "captured_gradients") and self.captured_gradients:
            grad_metrics = self.captured_gradients.copy()
        else:
            grad_metrics = {
                "op_logits_mean": 0.0,
            }
            for op_name in OP_NAMES:
                grad_metrics[f"grad/op/{op_name}"] = 0.0

        extra_vals = {}
        extra_vals.update(op_metrics)
        extra_vals.update(grad_metrics)

        return extra_vals

    def get_node_values_list(self, model) -> List[float]:
        """Get node values from the model as a list."""
        if model.config.dag_depth == 0:
            return []

        assert hasattr(model, "final_values"), "Model missing final_values attribute"
        assert model.final_values is not None, "final_values is None"
        assert isinstance(
            model.final_values, torch.Tensor
        ), "final_values must be a tensor"

        if model.final_values.dim() == 3:
            B, scratch_nodes, T = model.final_values.shape
            final_slot = (
                (model.config.dag_depth - 1) % scratch_nodes
                if model.config.dag_depth > 0
                else 1
            )
            final_values = model.final_values[:, final_slot, :]
            return [val.mean().item() for val in final_values.transpose(0, 1)]
        else:
            return [val.mean().item() for val in model.final_values.transpose(0, 1)]

    def get_detailed_node_values(self, model) -> dict:
        """Get detailed node values from the model showing all scratch nodes per token."""
        if model.config.dag_depth == 0:
            return {}

        assert hasattr(model, "final_values"), "Model missing final_values attribute"
        assert model.final_values is not None, "final_values is None"
        assert isinstance(
            model.final_values, torch.Tensor
        ), "final_values must be a tensor"
        assert (
            model.final_values.dim() == 3
        ), f"Expected 3D tensor, got {model.final_values.dim()}D"

        B, scratch_nodes, T = model.final_values.shape

        values_per_token = []
        aggregated_per_token = []

        for t in range(T):
            token_values = []
            for s in range(scratch_nodes):
                value = model.final_values[:, s, t].mean().item()
                token_values.append(value)

            values_per_token.append(token_values)
            aggregated_per_token.append(sum(token_values) / len(token_values))

        return {
            "values_per_token": values_per_token,
            "aggregated_per_token": aggregated_per_token,
            "scratch_nodes": scratch_nodes,
            "sequence_length": T,
            "batch_size": B,
        }

    def format_console_logging(self, model, decode_fn=None, input_ids=None) -> None:
        """Print formatted logging information to console."""
        if model.config.dag_depth == 0:
            return

        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(
            model.dag, "plan_predictor"
        ), "DAG missing plan_predictor attribute"
        assert hasattr(
            model.dag, "num_scratch_nodes"
        ), "DAG missing scratch_nodes attribute"
        assert model.dag.num_scratch_nodes is not None, "dag.num_scratch_nodes is None"
        assert model.dag.num_scratch_nodes > 0, "dag.num_scratch_nodes must be positive"
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

        print(
            "Initial node values per token position (converted to real values) for prompt sample:"
        )
        scratch_nodes = detailed_values["scratch_nodes"]

        token_texts = []
        if decode_fn is not None and input_ids is not None:
            batch_tokens = input_ids[0]
            for i, token_id in enumerate(batch_tokens):
                try:
                    token_text = decode_fn([token_id.item()])
                    token_text = repr(token_text)
                    token_texts.append(token_text)
                except Exception as e:
                    token_texts.append(f"<decode_error: {e}>")

        for t, token_values in enumerate(detailed_values["values_per_token"]):
            token_values = [math.exp(val) for val in token_values]

            if token_texts and t < len(token_texts):
                token_info = f"Token {t} {token_texts[t]}"
            else:
                token_info = f"Token {t}"

            if scratch_nodes > 1:
                values_str = ", ".join(
                    [f"{val:.4f}" for _, val in enumerate(token_values)]
                )
                print(f"  {token_info}: [{values_str}]")
            else:
                print(f"  {token_info}: {token_values[0]:.4f}")

    def get_wandb_logging_dict(self, model, base_dict: Optional[Dict] = None) -> Dict:
        """Get dictionary suitable for wandb logging."""
        if model.config.dag_depth == 0:
            return {}

        if base_dict is None:
            base_dict = {}

        log_dict = dict(base_dict)
        log_dict.update(self.get_extra_vals(model))

        return log_dict

    def _setup_op_grad_tracking(self, model):
        """Set up operation gradient tracking using the saved tensor from forward pass."""
        if model.config.dag_depth == 0:
            return

        assert hasattr(model, "dag"), "Model missing dag attribute"
        assert hasattr(
            model.dag, "plan_predictor"
        ), "DAG missing plan_predictor attribute"
        assert hasattr(
            model.dag.plan_predictor, "last_operation_probs"
        ), "Plan predictor missing last_operation_probs"
        assert (
            model.dag.plan_predictor.last_operation_probs is not None
        ), "last_operation_probs is None"

        if model.dag.plan_predictor.last_operation_probs.requires_grad:
            self.op_grad_hook = (
                model.dag.plan_predictor.last_operation_probs.register_hook(
                    self._op_grad_hook_fn
                )
            )
        else:
            self.op_grad_hook = None

    def _op_grad_hook_fn(self, grad):
        """Hook function to capture gradients of operation probabilities."""
        if grad is None:
            self.captured_gradients["op_logits_mean"] = 0.0
            for op_name in OP_NAMES:
                self.captured_gradients[f"grad/op/{op_name}"] = 0.0
            return
        grad_mean = grad.detach().mean().item()
        op_grads = grad.detach().mean(dim=(0, 1, 2)).cpu().numpy()

        self.captured_gradients["op_logits_mean"] = grad_mean
        for i, op_name in enumerate(OP_NAMES):
            self.captured_gradients[f"grad/op/{op_name}"] = float(op_grads[i])

    def _dag_output_grad_hook_fn(self, grad):
        """Hook function to capture gradients of DAG output."""
        if grad is None:
            self.captured_gradients["grad/dag_output_norm"] = 0.0
            self.captured_gradients["grad/dag_output_mean"] = 0.0
            self.captured_gradients["grad/dag_output_std"] = 0.0
            return
        grad_norm = grad.detach().norm().item()
        grad_mean = grad.detach().mean().item()
        grad_std = grad.detach().std().item()

        self.captured_gradients["grad/dag_output_norm"] = grad_norm
        self.captured_gradients["grad/dag_output_mean"] = grad_mean
        self.captured_gradients["grad/dag_output_std"] = grad_std

    def _dag_final_values_grad_hook_fn(self, grad):
        """Hook function to capture gradients of DAG final values."""
        if grad is None:
            self.captured_gradients["grad/dag_final_values_norm"] = 0.0
            self.captured_gradients["grad/dag_final_values_mean"] = 0.0
            return
        grad_norm = grad.detach().norm().item()
        grad_mean = grad.detach().mean().item()

        self.captured_gradients["grad/dag_final_values_norm"] = grad_norm
        self.captured_gradients["grad/dag_final_values_mean"] = grad_mean

    def _orig_hidden_grad_hook_fn(self, grad):
        """Hook function to capture gradients of original hidden states."""
        if grad is None:
            self.captured_gradients["grad/orig_hidden_mean"] = 0.0
            return
        grad_mean = grad.detach().mean().item()
        self.captured_gradients["grad/orig_hidden_mean"] = grad_mean

    def _gate_w_d_grad_hook_fn(self, grad):
        """Hook function to capture gradients of gate_w_d."""
        if grad is None:
            self.captured_gradients["grad/gate_w_d"] = 0.0
            return
        grad_norm = grad.detach().norm().item()
        self.captured_gradients["grad/gate_w_d"] = grad_norm

    def _gate_w_o_grad_hook_fn(self, grad):
        """Hook function to capture gradients of gate_w_o."""
        if grad is None:
            self.captured_gradients["grad/gate_w_o"] = 0.0
            return
        grad_norm = grad.detach().norm().item()
        self.captured_gradients["grad/gate_w_o"] = grad_norm

    def _collect_op_metrics(self, model) -> Dict[str, float]:
        """Compute op-selection sharpness, diversity and mutual information."""

        if not hasattr(model.dag.plan_predictor, "last_operation_probs"):
            raise RuntimeError("plan_predictor missing 'last_operation_probs'")
        if model.dag.plan_predictor.last_operation_probs is None:
            raise RuntimeError("'last_operation_probs' is None")

        probs = model.dag.plan_predictor.last_operation_probs.detach()
        n_ops = probs.size(-1)
        eps = 1e-12

        p_token = probs.reshape(-1, n_ops).float().clamp_min(eps)
        p_mean = p_token.mean(0)

        H_op = -torch.sum(p_mean * p_mean.log())
        H_cond = -torch.mean(
            torch.sum(p_token * p_token.log(), dim=1)
        )

        log_n = math.log(n_ops)
        op_diversity = (H_op / log_n).clamp(0.0, 1.0).item()
        op_sharpness = (1.0 - H_cond / log_n).clamp(0.0, 1.0).item()
        op_mi = ((H_op - H_cond) / log_n).clamp(0.0, 1.0).item()

        return {
            "op_diversity": op_diversity,
            "op_sharpness": op_sharpness,
            "op_mutual_info": op_mi,
        }
