#!/usr/bin/env python
"""
Tests for DAG predictor pretraining functionality.

This module tests the DAG predictor training components including
configuration management, loss functions, model setup, and training integration.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dag_model import GPT, GPTConfig
from data.dagset.streaming import create_dag_structure_dataloaders
from train_predictor import (DAGTrainConfig, ShallowAttentionConfig,
                             ShallowAttentionDAGPredictor, _all_tensors,
                             _safe_torch_save, apply_overrides,
                             clean_previous_checkpoints,
                             compute_dag_structure_loss, evaluate_dag_model,
                             find_latest_checkpoint, get_checkpoint_filename,
                             get_lr, load_config_file, update_config)


class TestDAGTrainConfig(unittest.TestCase):
    """Test configuration management for DAG training."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = DAGTrainConfig()
        self.assertEqual(cfg.name, "dag_pretrain")
        self.assertEqual(cfg.dag_depth, 4)
        self.assertEqual(cfg.n_embd, 768)
        self.assertEqual(cfg.learning_rate, 3e-4)
        self.assertEqual(cfg.sign_loss_weight, 1.0)
        self.assertEqual(cfg.log_loss_weight, 1.0)
        self.assertEqual(cfg.op_loss_weight, 1.0)

    def test_config_attributes(self):
        """Test that config has all required attributes."""
        cfg = DAGTrainConfig()
        required_attrs = [
            "name",
            "dag_depth",
            "n_embd",
            "n_head",
            "n_layer",
            "sequence_length",
            "batch_size",
            "learning_rate",
            "max_iters",
            "sign_loss_weight",
            "log_loss_weight",
            "op_loss_weight",
            "max_dag_depth",
            "train_examples_per_batch",
            "val_examples_per_batch",
            "gradient_accumulation_steps",
            "dropout",
            "bias",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
            "decay_lr",
            "warmup_iters",
            "lr_decay_iters",
            "min_lr",
            "backend",
            "dtype",
            "compile",
            "keep_alive",
            "check_nans",
            "note",
            "train_seed",
            "val_seed",
            "eval_interval",
            "log_interval",
            "eval_iters",
            "eval_only",
            "always_save_checkpoint",
            "clear_previous_checkpoints",
            "init_from",
        ]

        for attr in required_attrs:
            self.assertTrue(
                hasattr(cfg, attr), f"Config missing required attribute: {attr}"
            )


class TestShallowAttentionConfig(unittest.TestCase):
    """Test configuration for shallow attention model."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = ShallowAttentionConfig()
        self.assertEqual(cfg.vocab_size, 50304)
        self.assertEqual(cfg.n_embd, 768)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.dag_depth, 4)
        self.assertEqual(cfg.sequence_length, 512)
        self.assertEqual(cfg.dropout, 0.0)
        self.assertFalse(cfg.bias)

    def test_config_attributes(self):
        """Test that config has all required attributes."""
        cfg = ShallowAttentionConfig()
        required_attrs = [
            "vocab_size",
            "n_embd",
            "n_head",
            "dropout",
            "bias",
            "dag_depth",
            "sequence_length",
            "softmax_temperature",
        ]

        for attr in required_attrs:
            self.assertTrue(
                hasattr(cfg, attr), f"Config missing required attribute: {attr}"
            )


class TestShallowAttentionDAGPredictor(unittest.TestCase):
    """Test the new shallow attention DAG predictor model."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.config = ShallowAttentionConfig(
            vocab_size=1000,
            n_embd=64,
            n_head=4,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            sequence_length=32,
            softmax_temperature=20.0,
        )

    def test_model_creation(self):
        """Test model creation with various configurations."""
        model = ShallowAttentionDAGPredictor(self.config)

        # Test basic properties
        self.assertIsInstance(model, ShallowAttentionDAGPredictor)
        self.assertEqual(model.config.dag_depth, 2)
        self.assertEqual(model.config.n_embd, 64)

        # Test model has required components
        self.assertTrue(hasattr(model, "wte"))  # Token embeddings
        self.assertTrue(hasattr(model, "wpe"))  # Position embeddings
        self.assertTrue(hasattr(model, "attention_block"))  # Shallow attention
        self.assertTrue(hasattr(model, "dag_predictor"))  # DAG predictor

        # Test parameter count
        param_count = model.get_num_params()
        self.assertGreater(param_count, 0)

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        model = ShallowAttentionDAGPredictor(self.config)
        model.eval()

        batch_size = 2
        seq_len = self.config.sequence_length
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            # Test without internal state
            outputs = model(input_ids, return_internal_state=False)
            self.assertEqual(len(outputs), 3)

            pred_sgn, pred_log, pred_ops = outputs
            expected_nodes = self.config.dag_depth + 1

            self.assertEqual(pred_sgn.shape, (batch_size, seq_len, expected_nodes))
            self.assertEqual(pred_log.shape, (batch_size, seq_len, expected_nodes))
            self.assertEqual(
                pred_ops.shape, (batch_size, seq_len, self.config.dag_depth, 5)
            )

            # Test with internal state
            outputs_with_state = model(input_ids, return_internal_state=True)
            self.assertEqual(len(outputs_with_state), 4)

            pred_sgn2, pred_log2, pred_ops2, internal_state = outputs_with_state

            # Check that predictions are the same
            self.assertTrue(torch.allclose(pred_sgn, pred_sgn2, atol=1e-6))
            self.assertTrue(torch.allclose(pred_log, pred_log2, atol=1e-6))
            self.assertTrue(torch.allclose(pred_ops, pred_ops2, atol=1e-6))

            # Check internal state structure
            expected_internal_keys = [
                "initial_value_hidden",
                "dag_structure_hidden",
                "cross_output",
                "initial_values_raw",
                "operation_logits_raw",
                "sign_logits",
                "mag_logits",
                "operation_logits",
            ]
            self.assertEqual(set(internal_state.keys()), set(expected_internal_keys))

    def test_forward_pass_values(self):
        """Test forward pass produces reasonable values."""
        model = ShallowAttentionDAGPredictor(self.config)
        model.eval()

        batch_size = 2
        seq_len = self.config.sequence_length
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            pred_sgn, pred_log, pred_ops = model(input_ids, return_internal_state=False)

            # Signs should be in [-1, 1] range (tanh output)
            self.assertTrue((pred_sgn >= -1).all())
            self.assertTrue((pred_sgn <= 1).all())

            # Log magnitudes should be non-negative
            self.assertTrue((pred_log >= 0).all())

            # Operation probabilities should sum to 1 and be non-negative
            self.assertTrue((pred_ops >= 0).all())
            self.assertTrue((pred_ops <= 1).all())
            op_sums = pred_ops.sum(dim=-1)
            self.assertTrue(
                torch.allclose(op_sums, torch.ones_like(op_sums), atol=1e-5)
            )

            # All outputs should be finite
            self.assertTrue(torch.isfinite(pred_sgn).all())
            self.assertTrue(torch.isfinite(pred_log).all())
            self.assertTrue(torch.isfinite(pred_ops).all())

    def test_different_sequence_lengths(self):
        """Test model works with different sequence lengths."""
        model = ShallowAttentionDAGPredictor(self.config)
        model.eval()

        batch_size = 2
        test_lengths = [16, 32, 64]  # Test different lengths <= config.sequence_length

        for seq_len in test_lengths:
            if seq_len <= self.config.sequence_length:
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))

                with torch.no_grad():
                    pred_sgn, pred_log, pred_ops = model(
                        input_ids, return_internal_state=False
                    )

                    expected_nodes = self.config.dag_depth + 1
                    self.assertEqual(
                        pred_sgn.shape, (batch_size, seq_len, expected_nodes)
                    )
                    self.assertEqual(
                        pred_log.shape, (batch_size, seq_len, expected_nodes)
                    )
                    self.assertEqual(
                        pred_ops.shape, (batch_size, seq_len, self.config.dag_depth, 5)
                    )

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = ShallowAttentionDAGPredictor(self.config)
        model.train()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        pred_sgn, pred_log, pred_ops = model(input_ids, return_internal_state=False)

        # Create dummy loss
        loss = pred_sgn.mean() + pred_log.mean() + pred_ops.mean()

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
                )


class TestConfigManagement(unittest.TestCase):
    """Test configuration loading and manipulation functions."""

    def test_load_config_file(self):
        """Test loading configuration from a Python file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
# Test configuration
dag_depth = 6
learning_rate = 1e-3
batch_size = 16
n_embd = 512
"""
            )
            temp_path = f.name

        try:
            config_data = load_config_file(temp_path)
            self.assertEqual(config_data["dag_depth"], 6)
            self.assertEqual(config_data["learning_rate"], 1e-3)
            self.assertEqual(config_data["batch_size"], 16)
            self.assertEqual(config_data["n_embd"], 512)
        finally:
            os.unlink(temp_path)

    def test_update_config(self):
        """Test updating configuration with new values."""
        cfg = DAGTrainConfig()
        original_depth = cfg.dag_depth
        original_lr = cfg.learning_rate

        update_data = {"dag_depth": 8, "learning_rate": 5e-4}
        update_config(cfg, update_data)

        self.assertEqual(cfg.dag_depth, 8)
        self.assertEqual(cfg.learning_rate, 5e-4)
        self.assertNotEqual(cfg.dag_depth, original_depth)
        self.assertNotEqual(cfg.learning_rate, original_lr)

    def test_apply_overrides(self):
        """Test applying command-line overrides."""
        cfg = DAGTrainConfig()
        overrides = ["--dag_depth=3", "--learning_rate=2e-4"]

        apply_overrides(cfg, overrides)

        self.assertEqual(cfg.dag_depth, 3)
        self.assertEqual(cfg.learning_rate, 2e-4)


class TestLossFunctions(unittest.TestCase):
    """Test loss computation functions."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.cfg = DAGTrainConfig()
        self.cfg.dag_depth = 2
        self.cfg.sign_loss_weight = 1.0
        self.cfg.log_loss_weight = 1.0
        self.cfg.op_loss_weight = 1.0

    def test_compute_dag_structure_loss_shapes(self):
        """Test loss computation with various tensor shapes."""
        B, T = 2, 8
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = 5

        # Create test tensors
        pred_sgn = torch.tanh(torch.randn(B, T, num_nodes))
        pred_log = torch.abs(torch.randn(B, T, num_nodes))
        pred_ops = torch.softmax(torch.randn(B, T, depth, n_ops), dim=-1)

        target_sgn = torch.randn(B, T, num_nodes)
        target_log = torch.abs(torch.randn(B, T, num_nodes))
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1  # One-hot

        target_depths = torch.tensor([depth] * B)

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            target_depths,
            self.cfg,
        )

        # Check return format
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
        self.assertEqual(set(losses.keys()), expected_keys)

        # Check loss values are reasonable
        for key, loss in losses.items():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss.item(), 0.0)

    def test_compute_dag_structure_loss_with_internal_state(self):
        """Test loss computation with internal states."""
        B, T = 2, 8
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = 5
        H = self.cfg.n_embd

        # Create test tensors
        pred_sgn = torch.tanh(torch.randn(B, T, num_nodes))
        pred_log = torch.abs(torch.randn(B, T, num_nodes))
        pred_ops = torch.softmax(torch.randn(B, T, depth, n_ops), dim=-1)

        target_sgn = torch.randn(B, T, num_nodes)
        target_log = torch.abs(torch.randn(B, T, num_nodes))
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        target_depths = torch.tensor([depth] * B)

        # Create internal state
        internal_state = {
            "initial_value_hidden": torch.randn(B, T, H),
            "dag_structure_hidden": torch.randn(B, T, H),
            "cross_output": torch.randn(B, T, H),
            "initial_values_raw": torch.randn(B, T, 2 * num_nodes),
            "operation_logits_raw": torch.randn(B, T, depth * n_ops),
            "sign_logits": torch.randn(B, T, num_nodes),
            "mag_logits": torch.randn(B, T, num_nodes),
            "operation_logits": torch.randn(B, T, depth, n_ops),
        }

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            target_depths,
            self.cfg,
            internal_state,
        )

        # Should only have the basic loss components (no internal regularization losses)
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}

        self.assertEqual(set(losses.keys()), expected_keys)

        # Check all losses are reasonable
        for key, loss in losses.items():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss.item(), 0.0)

    def test_compute_dag_structure_loss_weights(self):
        """Test loss weighting works correctly."""
        B, T = 2, 4
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = 5

        pred_sgn = torch.tanh(torch.randn(B, T, num_nodes))
        pred_log = torch.abs(torch.randn(B, T, num_nodes))
        pred_ops = torch.softmax(torch.randn(B, T, depth, n_ops), dim=-1)

        target_sgn = torch.randn(B, T, num_nodes)
        target_log = torch.abs(torch.randn(B, T, num_nodes))
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        target_depths = torch.tensor([depth] * B)

        # Test with different weights
        cfg_weighted = DAGTrainConfig()
        cfg_weighted.dag_depth = self.cfg.dag_depth
        cfg_weighted.sign_loss_weight = 2.0
        cfg_weighted.log_loss_weight = 0.5
        cfg_weighted.op_loss_weight = 1.5

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            target_depths,
            cfg_weighted,
        )

        # Verify total loss incorporates weights
        expected_total = (
            cfg_weighted.sign_loss_weight * losses["sign_loss"]
            + cfg_weighted.log_loss_weight * losses["log_loss"]
            + cfg_weighted.op_loss_weight * losses["op_loss"]
        )
        self.assertTrue(torch.allclose(losses["total_loss"], expected_total, atol=1e-5))

    def test_compute_dag_structure_loss_perfect_prediction(self):
        """Test loss with perfect predictions."""
        B, T = 2, 4
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = 5

        # Create identical predictions and targets
        target_sgn = torch.randn(B, T, num_nodes)
        target_log = torch.abs(torch.randn(B, T, num_nodes))
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        # Perfect predictions = targets
        pred_sgn = target_sgn.clone()
        pred_log = target_log.clone()
        pred_ops = target_ops.clone()

        target_depths = torch.tensor([depth] * B)

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            target_depths,
            self.cfg,
        )

        # Losses should be very small for perfect predictions
        self.assertLess(losses["sign_loss"].item(), 1e-6)
        self.assertLess(losses["log_loss"].item(), 1e-6)
        self.assertLess(losses["op_loss"].item(), 1e-6)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_lr_warmup(self):
        """Test learning rate warmup schedule."""
        cfg = DAGTrainConfig()
        cfg.warmup_iters = 1000
        cfg.learning_rate = 1e-3
        cfg.min_lr = 1e-5

        # During warmup
        lr_warmup = get_lr(500, cfg=cfg)
        expected_warmup = cfg.learning_rate * (500 + 1) / (cfg.warmup_iters + 1)
        self.assertAlmostEqual(lr_warmup, expected_warmup, places=6)

        # At end of warmup
        lr_end_warmup = get_lr(cfg.warmup_iters, cfg=cfg)
        self.assertAlmostEqual(lr_end_warmup, cfg.learning_rate, places=6)

    def test_get_lr_decay(self):
        """Test learning rate decay schedule."""
        cfg = DAGTrainConfig()
        cfg.warmup_iters = 1000
        cfg.lr_decay_iters = 5000
        cfg.learning_rate = 1e-3
        cfg.min_lr = 1e-5

        # After decay period
        lr_final = get_lr(cfg.lr_decay_iters + 1000, cfg=cfg)
        self.assertEqual(lr_final, cfg.min_lr)

        # During decay period
        lr_decay = get_lr(3000, cfg=cfg)
        self.assertGreater(lr_decay, cfg.min_lr)
        self.assertLess(lr_decay, cfg.learning_rate)

    def test_get_checkpoint_filename(self):
        """Test checkpoint filename generation."""
        cfg = DAGTrainConfig()
        cfg.name = "test_model"
        iter_num = 1500

        filename = get_checkpoint_filename(cfg, iter_num)
        self.assertEqual(filename, "dag_ckpt_test_model_001500.pt")

    def test_all_tensors(self):
        """Test _all_tensors utility function."""
        # All tensors
        state_all_tensors = {
            "layer1": torch.randn(5, 3),
            "layer2": {"weight": torch.randn(10, 5), "bias": torch.randn(10)},
        }
        self.assertTrue(_all_tensors(state_all_tensors))

        # Mixed types
        state_mixed = {
            "layer1": torch.randn(5, 3),
            "layer2": {"weight": torch.randn(10, 5), "bias": 0.1},  # Not a tensor
        }
        self.assertFalse(_all_tensors(state_mixed))


class TestModelSetup(unittest.TestCase):
    """Test model setup and initialization."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_model_creation_with_dag(self):
        """Test creating model with DAG components."""
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.n_layer = 2
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.sequence_length,
            bias=cfg.bias,
            vocab_size=1000,
            dropout=cfg.dropout,
            dag_depth=cfg.dag_depth,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Check that model has DAG components
        self.assertTrue(hasattr(model, "dag"))
        self.assertIsNotNone(model.dag)
        self.assertTrue(hasattr(model.dag, "plan_predictor"))

    def test_parameter_freezing(self):
        """Test freezing non-DAG parameters."""
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.n_layer = 2
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.sequence_length,
            bias=cfg.bias,
            vocab_size=1000,
            dropout=cfg.dropout,
            dag_depth=cfg.dag_depth,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Freeze non-DAG parameters
        for name, param in model.named_parameters():
            if "dag.plan_predictor" not in name:
                param.requires_grad = False

        # Check freezing worked
        dag_params = [p for n, p in model.named_parameters() if p.requires_grad]
        all_params = list(model.parameters())

        self.assertGreater(len(all_params), len(dag_params))
        self.assertGreater(len(dag_params), 0)

    def test_model_forward_with_dag_predictor(self):
        """Test forward pass through DAG predictor."""
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.n_layer = 2
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.sequence_length,
            bias=cfg.bias,
            vocab_size=1000,
            dropout=cfg.dropout,
            dag_depth=cfg.dag_depth,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.eval()

        batch_size = 2
        input_tokens = torch.randint(0, 1000, (batch_size, cfg.sequence_length))

        with torch.no_grad():
            # Get hidden states
            pos = torch.arange(cfg.sequence_length)
            emb = model.transformer.wte(input_tokens) + model.transformer.wpe(pos)
            hidden = model.transformer.drop(emb)
            for block in model.transformer.h:
                hidden = block(hidden)
            hidden = model.transformer.ln_f(hidden)

            # DAG predictor forward pass
            pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden, hidden)

            # Verify output shapes
            expected_nodes = cfg.dag_depth + 1
            self.assertEqual(
                pred_sgn.shape, (batch_size, cfg.sequence_length, expected_nodes)
            )
            self.assertEqual(
                pred_log.shape, (batch_size, cfg.sequence_length, expected_nodes)
            )
            self.assertEqual(
                pred_ops.shape, (batch_size, cfg.sequence_length, cfg.dag_depth, 5)
            )

            # Verify outputs are reasonable
            self.assertTrue(torch.isfinite(pred_sgn).all())
            self.assertTrue(torch.isfinite(pred_log).all())
            self.assertTrue(torch.isfinite(pred_ops).all())
            self.assertTrue(
                (pred_log >= 0).all()
            )  # Log magnitudes should be non-negative


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint saving and loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = DAGTrainConfig()
        self.cfg.name = "test_checkpoint"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_torch_save(self):
        """Test safe checkpoint saving."""
        test_data = {
            "model": {"param1": torch.randn(5, 3), "param2": torch.randn(10)},
            "optimizer": {"state": {}, "param_groups": []},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        save_path = Path(self.temp_dir) / "test_checkpoint.pt"

        # Test successful save
        _safe_torch_save(test_data, save_path)
        self.assertTrue(save_path.exists())

        # Load and verify
        loaded_data = torch.load(save_path)
        self.assertEqual(loaded_data["iter_num"], 100)
        self.assertEqual(loaded_data["best_val_loss"], 0.5)
        self.assertTrue(
            torch.equal(loaded_data["model"]["param1"], test_data["model"]["param1"])
        )

    def test_clean_previous_checkpoints(self):
        """Test cleaning previous checkpoints."""

        cfg = DAGTrainConfig()
        cfg.name = "test_clean"
        cfg.clear_previous_checkpoints = True

        # Create some fake checkpoint files
        checkpoint_dir = Path(self.temp_dir)
        (checkpoint_dir / "dag_ckpt_test_clean_000100.pt").touch()
        (checkpoint_dir / "dag_ckpt_test_clean_000200.pt").touch()
        (
            checkpoint_dir / "dag_ckpt_other_000100.pt"
        ).touch()  # Different name, shouldn't be removed

        with patch("train_predictor.CHECKPOINT_DIR", self.temp_dir):
            clean_previous_checkpoints(cfg)

        # Check that only the matching checkpoints were removed
        self.assertFalse((checkpoint_dir / "dag_ckpt_test_clean_000100.pt").exists())
        self.assertFalse((checkpoint_dir / "dag_ckpt_test_clean_000200.pt").exists())
        self.assertTrue((checkpoint_dir / "dag_ckpt_other_000100.pt").exists())

    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint."""
        cfg = DAGTrainConfig()
        cfg.name = "test_find"

        checkpoint_dir = Path(self.temp_dir)

        # Create checkpoint files with different iteration numbers
        (checkpoint_dir / "dag_ckpt_test_find_000100.pt").touch()
        (checkpoint_dir / "dag_ckpt_test_find_000500.pt").touch()
        (checkpoint_dir / "dag_ckpt_test_find_000300.pt").touch()

        with patch("train_predictor.CHECKPOINT_DIR", self.temp_dir):
            latest = find_latest_checkpoint(cfg)

        self.assertIsNotNone(latest)
        self.assertEqual(latest.name, "dag_ckpt_test_find_000500.pt")


class TestIntegration(unittest.TestCase):
    """Integration tests for training components."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.cfg = DAGTrainConfig()
        self.cfg.dag_depth = 2
        self.cfg.n_layer = 2
        self.cfg.n_head = 2
        self.cfg.n_embd = 32
        self.cfg.batch_size = 2
        self.cfg.sequence_length = 16

    def test_evaluate_shallow_attention_model(self):
        """Test evaluation function with new shallow attention model."""
        # Create shallow attention model
        model_config = ShallowAttentionConfig(
            vocab_size=1000,
            n_embd=self.cfg.n_embd,
            n_head=self.cfg.n_head,
            dropout=self.cfg.dropout,
            bias=self.cfg.bias,
            dag_depth=self.cfg.dag_depth,
            sequence_length=self.cfg.sequence_length,
            softmax_temperature=20.0,
        )
        model = ShallowAttentionDAGPredictor(model_config)
        model.eval()

        # Create mock data loader
        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=self.cfg.batch_size,
            val_batch_size=self.cfg.batch_size,
            max_depth=self.cfg.dag_depth,
            train_seed=42,
            val_seed=43,
        )

        # Test evaluation (should not crash)
        device = "cpu"
        ctx = torch.no_grad()

        try:
            val_losses = evaluate_dag_model(
                model, val_loader, device, ctx, self.cfg, eval_iters=2
            )

            # Verify return format - should include internal losses when using shallow attention model
            expected_base_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
            self.assertTrue(expected_base_keys.issubset(set(val_losses.keys())))

            for key, loss in val_losses.items():
                self.assertIsInstance(loss, float)
                self.assertTrue(np.isfinite(loss))
                self.assertGreaterEqual(loss, 0.0)

        except Exception as e:
            self.fail(f"Evaluation failed: {e}")

    def test_evaluate_dag_model_integration(self):
        """Test evaluation function integration with legacy GPT model."""
        # Create legacy GPT model for backward compatibility testing
        model_args = dict(
            n_layer=self.cfg.n_layer,
            n_head=self.cfg.n_head,
            n_embd=self.cfg.n_embd,
            block_size=self.cfg.sequence_length,
            bias=self.cfg.bias,
            vocab_size=1000,
            dropout=self.cfg.dropout,
            dag_depth=self.cfg.dag_depth,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.eval()

        # Create mock data loader
        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=self.cfg.batch_size,
            val_batch_size=self.cfg.batch_size,
            max_depth=self.cfg.dag_depth,
            train_seed=42,
            val_seed=43,
        )

        # Test evaluation with manual implementation (since GPT doesn't support return_internal_state)
        device = "cpu"

        try:
            # Manual evaluation logic for GPT model
            losses_accum = {
                "total_loss": 0.0,
                "sign_loss": 0.0,
                "log_loss": 0.0,
                "op_loss": 0.0,
            }
            num_batches = 2

            for i, (texts, structures) in enumerate(val_loader):
                if i >= num_batches:
                    break

                input_tokens = torch.randint(
                    0, 1000, (len(texts), self.cfg.sequence_length)
                )

                # Forward through GPT to get hidden states
                pos = torch.arange(self.cfg.sequence_length)
                emb = model.transformer.wte(input_tokens) + model.transformer.wpe(pos)
                hidden = model.transformer.drop(emb)
                for block in model.transformer.h:
                    hidden = block(hidden)
                hidden = model.transformer.ln_f(hidden)

                # Forward through DAG predictor (using old interface)
                pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden, hidden)

                # Prepare targets
                target_sgn = structures["initial_sgn"].to(device)
                target_log = structures["initial_log"].to(device)
                target_ops = structures["operation_probs"].to(device)
                target_depths = structures["depths"].to(device)

                # Reshape for compatibility
                pred_sgn_reshaped = pred_sgn.mean(
                    dim=1, keepdim=True
                )  # Average over sequence
                pred_log_reshaped = pred_log.mean(dim=1, keepdim=True)
                pred_ops_reshaped = pred_ops.mean(dim=1, keepdim=True)

                # Ensure compatible shapes by taking first nodes/operations
                min_nodes = min(pred_sgn_reshaped.size(-1), target_sgn.size(-1))
                min_depth = min(pred_ops_reshaped.size(-2), target_ops.size(-2))

                pred_sgn_compat = pred_sgn_reshaped[..., :min_nodes]
                pred_log_compat = pred_log_reshaped[..., :min_nodes]
                pred_ops_compat = pred_ops_reshaped[..., :min_depth, :]

                target_sgn_compat = target_sgn.unsqueeze(1)[..., :min_nodes]
                target_log_compat = target_log.unsqueeze(1)[..., :min_nodes]
                target_ops_compat = target_ops.unsqueeze(1)[..., :min_depth, :]

                # Compute losses
                batch_losses = compute_dag_structure_loss(
                    pred_sgn_compat,
                    pred_log_compat,
                    pred_ops_compat,
                    target_sgn_compat,
                    target_log_compat,
                    target_ops_compat,
                    target_depths,
                    self.cfg,
                )

                # Accumulate
                for key in losses_accum:
                    losses_accum[key] += batch_losses[key].item()

            # Average over batches
            val_losses = {k: v / num_batches for k, v in losses_accum.items()}

            # Verify return format
            expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
            self.assertEqual(set(val_losses.keys()), expected_keys)

            for key, loss in val_losses.items():
                self.assertIsInstance(loss, float)
                self.assertTrue(np.isfinite(loss))
                self.assertGreaterEqual(loss, 0.0)

        except Exception as e:
            self.fail(f"Evaluation failed: {e}")

    def test_training_step_integration(self):
        """Test a single training step integration with legacy GPT model."""
        # Create legacy GPT model
        model_args = dict(
            n_layer=self.cfg.n_layer,
            n_head=self.cfg.n_head,
            n_embd=self.cfg.n_embd,
            block_size=self.cfg.sequence_length,
            bias=self.cfg.bias,
            vocab_size=1000,
            dropout=self.cfg.dropout,
            dag_depth=self.cfg.dag_depth,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.train()

        # Freeze non-DAG parameters
        for name, param in model.named_parameters():
            if "dag.plan_predictor" not in name:
                param.requires_grad = False

        # Create optimizer for DAG parameters only
        dag_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(dag_params, lr=1e-4)

        # Create data loader
        train_loader, _ = create_dag_structure_dataloaders(
            train_batch_size=self.cfg.batch_size,
            val_batch_size=self.cfg.batch_size,
            max_depth=self.cfg.dag_depth,
            train_seed=42,
            val_seed=43,
        )

        # Get a batch
        texts, structures = next(train_loader)

        device = "cpu"
        target_sgn = structures["initial_sgn"].to(device)
        target_log = structures["initial_log"].to(device)
        target_ops = structures["operation_probs"].to(device)
        target_depths = structures["depths"].to(device)

        # Training step
        optimizer.zero_grad()

        # Forward pass
        batch_size = len(texts)
        input_tokens = torch.randint(0, 1000, (batch_size, self.cfg.sequence_length))

        pos = torch.arange(self.cfg.sequence_length)
        emb = model.transformer.wte(input_tokens) + model.transformer.wpe(pos)
        hidden = model.transformer.drop(emb)
        for block in model.transformer.h:
            hidden = block(hidden)
        hidden = model.transformer.ln_f(hidden)

        # DAG predictor forward
        pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden, hidden)

        # Reshape for compatibility
        pred_sgn_reshaped = pred_sgn.mean(dim=1, keepdim=True)  # Average over sequence
        pred_log_reshaped = pred_log.mean(dim=1, keepdim=True)
        pred_ops_reshaped = pred_ops.mean(dim=1, keepdim=True)

        # Ensure compatible shapes
        min_nodes = min(pred_sgn_reshaped.size(-1), target_sgn.size(-1))
        min_depth = min(pred_ops_reshaped.size(-2), target_ops.size(-2))

        pred_sgn_compat = pred_sgn_reshaped[..., :min_nodes]
        pred_log_compat = pred_log_reshaped[..., :min_nodes]
        pred_ops_compat = pred_ops_reshaped[..., :min_depth, :]

        target_sgn_compat = target_sgn.unsqueeze(1)[..., :min_nodes]
        target_log_compat = target_log.unsqueeze(1)[..., :min_nodes]
        target_ops_compat = target_ops.unsqueeze(1)[..., :min_depth, :]

        # Compute loss
        losses = compute_dag_structure_loss(
            pred_sgn_compat,
            pred_log_compat,
            pred_ops_compat,
            target_sgn_compat,
            target_log_compat,
            target_ops_compat,
            target_depths,
            self.cfg,
        )

        loss = losses["total_loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist
        grad_count = 0
        for param in dag_params:
            if param.grad is not None:
                grad_count += 1

        self.assertGreater(grad_count, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_shallow_attention_training_step(self):
        """Test training step with new shallow attention model."""
        # Create shallow attention model
        model_config = ShallowAttentionConfig(
            vocab_size=1000,
            n_embd=self.cfg.n_embd,
            n_head=self.cfg.n_head,
            dropout=self.cfg.dropout,
            bias=self.cfg.bias,
            dag_depth=self.cfg.dag_depth,
            sequence_length=self.cfg.sequence_length,
            softmax_temperature=20.0,
        )
        model = ShallowAttentionDAGPredictor(model_config)
        model.train()

        # Create optimizer for all parameters (no freezing needed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create data loader
        train_loader, _ = create_dag_structure_dataloaders(
            train_batch_size=self.cfg.batch_size,
            val_batch_size=self.cfg.batch_size,
            max_depth=self.cfg.dag_depth,
            train_seed=42,
            val_seed=43,
        )

        # Get a batch
        texts, structures = next(train_loader)

        device = "cpu"
        target_sgn = structures["initial_sgn"].to(device)
        target_log = structures["initial_log"].to(device)
        target_ops = structures["operation_probs"].to(device)
        target_depths = structures["depths"].to(device)

        # Training step
        optimizer.zero_grad()

        # Forward pass
        batch_size = len(texts)
        input_tokens = torch.randint(0, 1000, (batch_size, self.cfg.sequence_length))

        # Forward through shallow attention model with internal state
        pred_sgn, pred_log, pred_ops, internal_state = model(
            input_tokens, return_internal_state=True
        )

        # Reshape for compatibility
        pred_sgn_reshaped = pred_sgn.mean(dim=1, keepdim=True)  # Average over sequence
        pred_log_reshaped = pred_log.mean(dim=1, keepdim=True)
        pred_ops_reshaped = pred_ops.mean(dim=1, keepdim=True)

        # Ensure compatible shapes
        min_nodes = min(pred_sgn_reshaped.size(-1), target_sgn.size(-1))
        min_depth = min(pred_ops_reshaped.size(-2), target_ops.size(-2))

        pred_sgn_compat = pred_sgn_reshaped[..., :min_nodes]
        pred_log_compat = pred_log_reshaped[..., :min_nodes]
        pred_ops_compat = pred_ops_reshaped[..., :min_depth, :]

        target_sgn_compat = target_sgn.unsqueeze(1)[..., :min_nodes]
        target_log_compat = target_log.unsqueeze(1)[..., :min_nodes]
        target_ops_compat = target_ops.unsqueeze(1)[..., :min_depth, :]

        # Compute loss with internal state
        losses = compute_dag_structure_loss(
            pred_sgn_compat,
            pred_log_compat,
            pred_ops_compat,
            target_sgn_compat,
            target_log_compat,
            target_ops_compat,
            target_depths,
            self.cfg,
            internal_state,
        )

        loss = losses["total_loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1

        self.assertGreater(grad_count, 0)
        self.assertTrue(torch.isfinite(loss))

        # Verify we only have main prediction losses (no internal regularization)
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
        self.assertEqual(set(losses.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
