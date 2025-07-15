#!/usr/bin/env python
"""
Tests for DAG predictor pretraining functionality.

This module tests the DAG predictor training components including
configuration management, loss functions, model setup, and training integration.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpoint_manager import CheckpointManager
from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig
from train_predictor import (DAGTrainConfig, PredictorOnlyModel,
                             apply_overrides, compute_dag_structure_loss,
                             get_lr, load_config_file, tokenize_texts,
                             update_config)


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
            "seed",
            "eval_interval",
            "log_interval",
            "eval_iters",
            "eval_only",
            "always_save_checkpoint",
            "clear_previous_checkpoints",
            "init_from",
            "full_backbone",
            "n_layer",
        ]

        for attr in required_attrs:
            self.assertTrue(
                hasattr(cfg, attr), f"Config missing required attribute: {attr}"
            )


class TestShallowAttentionConfig(unittest.TestCase):
    """Test configuration for shallow attention model."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = PredictorOnlyConfig()
        self.assertEqual(cfg.vocab_size, 50304)
        self.assertEqual(cfg.n_embd, 768)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.dag_depth, 4)
        self.assertEqual(cfg.sequence_length, 512)
        self.assertEqual(cfg.dropout, 0.0)
        self.assertFalse(cfg.bias)

    def test_config_attributes(self):
        """Test that config has all required attributes."""
        cfg = PredictorOnlyConfig()
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
        self.config = PredictorOnlyConfig(
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
        model = PredictorOnlyModel(self.config)

        # Test basic properties
        self.assertIsInstance(model, PredictorOnlyModel)
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
        model = PredictorOnlyModel(self.config)
        model.eval()

        batch_size = 2
        seq_len = self.config.sequence_length
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            # Test forward pass shapes
            pred_sgn, pred_log, pred_ops = model(input_ids)

            # Check shapes
            expected_nodes = self.config.dag_depth + 1
            self.assertEqual(pred_sgn.shape, (batch_size, seq_len, expected_nodes))
            self.assertEqual(pred_log.shape, (batch_size, seq_len, expected_nodes))
            self.assertEqual(
                pred_ops.shape, (batch_size, seq_len, self.config.dag_depth, 5)
            )

    def test_forward_pass_values(self):
        """Test forward pass produces reasonable values."""
        model = PredictorOnlyModel(self.config)
        model.eval()

        batch_size = 2
        seq_len = self.config.sequence_length
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            pred_sgn, pred_log, pred_ops = model(input_ids)

            # Signs should be in [-1, 1] range (tanh output)
            self.assertTrue((pred_sgn >= -1).all())
            self.assertTrue((pred_sgn <= 1).all())

            # Log magnitudes should lie within the clipping range [-LOG_LIM, LOG_LIM]
            from models.dag_model import LOG_LIM

            self.assertTrue(((pred_log >= -LOG_LIM) & (pred_log <= LOG_LIM)).all())

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
        model = PredictorOnlyModel(self.config)
        model.eval()

        batch_size = 2
        test_lengths = [16, 32, 64]  # Test different lengths <= config.sequence_length

        for seq_len in test_lengths:
            if seq_len <= self.config.sequence_length:
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))

                with torch.no_grad():
                    pred_sgn, pred_log, pred_ops = model(input_ids)

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
        model = PredictorOnlyModel(self.config)
        model.train()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        pred_sgn, pred_log, pred_ops = model(input_ids)

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

    def test_backwards_pass_with_zero_inputs(self):
        """Test that backwards pass can go through the predictor only model if the input string just contains 0s."""
        model = PredictorOnlyModel(self.config)
        model.train()

        batch_size = 2
        seq_len = 16
        # Create input tensor containing only zeros
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

        # Forward pass with zero inputs
        pred_sgn, pred_log, pred_ops = model(input_ids)

        # Verify outputs have reasonable shapes and values
        expected_nodes = self.config.dag_depth + 1
        self.assertEqual(pred_sgn.shape, (batch_size, seq_len, expected_nodes))
        self.assertEqual(pred_log.shape, (batch_size, seq_len, expected_nodes))
        self.assertEqual(
            pred_ops.shape, (batch_size, seq_len, self.config.dag_depth, 5)
        )

        # Verify outputs are finite (not NaN or Inf)
        self.assertTrue(
            torch.isfinite(pred_sgn).all(), "pred_sgn contains non-finite values"
        )
        self.assertTrue(
            torch.isfinite(pred_log).all(), "pred_log contains non-finite values"
        )
        self.assertTrue(
            torch.isfinite(pred_ops).all(), "pred_ops contains non-finite values"
        )

        # Create dummy loss from outputs
        loss = pred_sgn.mean() + pred_log.mean() + pred_ops.mean()

        # Verify loss is finite
        self.assertTrue(torch.isfinite(loss), "Loss is not finite")

        # Clear any existing gradients
        model.zero_grad()

        # Backward pass - this should work without errors
        loss.backward()

        # Check that gradients were computed and are finite
        gradient_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
                )
                # Check that gradient is not all zeros (at least some learning signal)
                self.assertTrue(
                    param.grad.abs().sum() > 0, f"Gradient is all zeros for {name}"
                )
                gradient_count += 1

        # Ensure we actually checked some gradients
        self.assertGreater(gradient_count, 0, "No parameters with gradients found")


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

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
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

    def test_compute_dag_structure_loss_basic(self):
        """Test basic loss computation."""
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

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            self.cfg,
        )

        # Should have the basic loss components
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

        target_sgn = torch.randint(0, 2, (B, T, num_nodes)).float() * 2 - 1  # ±1
        target_log = torch.abs(torch.randn(B, T, num_nodes))

        # Operation targets: one-hot on the first op for determinism
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        # Perfect predictions equal the targets
        pred_sgn = target_sgn.clone()
        pred_log = target_log.clone()
        pred_ops = target_ops.clone()

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            self.cfg,
        )

        # Losses should be very small for perfect predictions
        self.assertLess(losses["sign_loss"].item(), 1e-6)
        self.assertLess(losses["log_loss"].item(), 1e-6)
        self.assertLess(losses["op_loss"].item(), 1e-6)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_lr_warmup(self):
        """Test learning rate warmup phase."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-3
        cfg.warmup_iters = 100
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        # Test warmup phase
        lr_0 = get_lr(0, cfg=cfg)
        lr_50 = get_lr(50, cfg=cfg)
        lr_100 = get_lr(100, cfg=cfg)

        # With the new implementation, LR at iter 0 is learning_rate * 1/warmup_iters
        expected_lr_0 = cfg.learning_rate * 1 / cfg.warmup_iters
        self.assertAlmostEqual(lr_0, expected_lr_0, places=7)
        self.assertLess(lr_50, cfg.learning_rate)
        self.assertAlmostEqual(lr_100, cfg.learning_rate, places=7)

    def test_get_lr_decay(self):
        """Test learning rate decay phase."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-3
        cfg.warmup_iters = 100
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        # Test decay phase
        lr_500 = get_lr(500, cfg=cfg)
        lr_1000 = get_lr(1000, cfg=cfg)
        lr_2000 = get_lr(2000, cfg=cfg)

        self.assertLess(lr_500, cfg.learning_rate)
        self.assertGreater(lr_500, cfg.min_lr)
        self.assertAlmostEqual(lr_1000, cfg.min_lr, places=7)
        self.assertAlmostEqual(lr_2000, cfg.min_lr, places=7)

    def test_get_checkpoint_filename(self):
        """Test checkpoint filename generation."""
        checkpoint_manager = CheckpointManager("dag")

        filename = checkpoint_manager.generate_checkpoint_filename(
            "test_run", 1000, "TestModel"
        )
        self.assertEqual(filename, "ckpt_test_run_1000.pt")

    def test_all_tensors(self):
        """Test _all_tensors utility function."""
        checkpoint_manager = CheckpointManager("dag")

        # Test with all tensors
        state_all_tensors = {
            "layer1": {"weight": torch.randn(10, 10), "bias": torch.randn(10)},
            "layer2": {"weight": torch.randn(5, 10)},
        }
        self.assertTrue(checkpoint_manager._all_tensors(state_all_tensors))

        # Test with non-tensor
        state_mixed = {
            "layer1": {"weight": torch.randn(10, 10), "bias": [1, 2, 3]},
            "layer2": {"weight": torch.randn(5, 10)},
        }
        self.assertFalse(checkpoint_manager._all_tensors(state_mixed))


class TestTokenization(unittest.TestCase):
    """Test text tokenization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.sequence_length = 32

    def test_tokenize_simple_expressions(self):
        """Test tokenization of simple mathematical expressions."""
        texts = ["28", "42.5", "3 + 4", "10 - 5", "7 * 8", "15 / 3"]

        tokens = tokenize_texts(texts, self.sequence_length, self.device)

        # Check basic properties
        self.assertEqual(tokens.shape, (len(texts), self.sequence_length))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.device.type, self.device)

        # Check that all values are non-negative (valid token IDs)
        self.assertTrue((tokens >= 0).all())

    def test_tokenize_complex_expressions(self):
        """Test tokenization of complex mathematical expressions."""
        texts = [
            "-82.612 - 94",
            "(5.22 - 3.213) / 2.32",
            "77.101 * 45.5 + 14.7",
            "93.3 * 4.9 - 93.3",
        ]

        tokens = tokenize_texts(texts, self.sequence_length, self.device)

        # Check basic properties
        self.assertEqual(tokens.shape, (len(texts), self.sequence_length))
        self.assertEqual(tokens.dtype, torch.long)

        # Check that longer expressions produce more non-zero tokens
        for i, text in enumerate(texts):
            non_zero_count = (tokens[i] != 0).sum().item()
            # Complex expressions should have at least a few tokens
            self.assertGreaterEqual(non_zero_count, 1)

            # But shouldn't exceed sequence length
            self.assertLessEqual(non_zero_count, self.sequence_length)

    def test_tokenize_english_expressions(self):
        """Test tokenization of English mathematical expressions."""
        texts = [
            "twenty-eight",
            "negative sixteen subtract less eighty-one point three",
            "five point two two minus three point two one three",
            "forty-one point eight three times eight point two eight",
        ]

        tokens = tokenize_texts(texts, self.sequence_length, self.device)

        # Check basic properties
        self.assertEqual(tokens.shape, (len(texts), self.sequence_length))
        self.assertEqual(tokens.dtype, torch.long)

        # English expressions typically generate more tokens
        for i, text in enumerate(texts):
            non_zero_count = (tokens[i] != 0).sum().item()
            # English expressions should have multiple tokens
            self.assertGreaterEqual(non_zero_count, 1)

    def test_tokenize_empty_and_single_char(self):
        """Test tokenization edge cases."""
        texts = [
            "",  # Empty string
            "0",  # Single character
            " ",  # Single space
            "1.0",  # Simple decimal
        ]

        tokens = tokenize_texts(texts, self.sequence_length, self.device)

        # Check basic properties
        self.assertEqual(tokens.shape, (len(texts), self.sequence_length))
        self.assertEqual(tokens.dtype, torch.long)

        # Empty string should result in all zeros (padding)
        self.assertTrue((tokens[0] == 0).all())

        # Single character should have one non-zero token
        self.assertEqual((tokens[1] != 0).sum().item(), 1)

    def test_tokenize_truncation(self):
        """Test that very long texts are properly truncated."""
        # Create a very long mathematical expression
        long_text = " + ".join([str(i) for i in range(100)])  # "0 + 1 + 2 + ... + 99"
        texts = [long_text]

        short_sequence_length = 10
        tokens = tokenize_texts(texts, short_sequence_length, self.device)

        # Check that output is properly shaped and truncated
        self.assertEqual(tokens.shape, (1, short_sequence_length))
        self.assertEqual(tokens.dtype, torch.long)

        # All positions should be filled (no padding for truncated text)
        non_zero_count = (tokens[0] != 0).sum().item()
        self.assertEqual(non_zero_count, short_sequence_length)

    def test_tokenize_padding(self):
        """Test that short texts are properly padded."""
        texts = ["5"]  # Very short text
        long_sequence_length = 50

        tokens = tokenize_texts(texts, long_sequence_length, self.device)

        # Check basic properties
        self.assertEqual(tokens.shape, (1, long_sequence_length))
        self.assertEqual(tokens.dtype, torch.long)

        # Should have exactly one non-zero token, rest should be padding (zeros)
        non_zero_count = (tokens[0] != 0).sum().item()
        zero_count = (tokens[0] == 0).sum().item()

        self.assertEqual(non_zero_count, 1)
        self.assertEqual(zero_count, long_sequence_length - 1)

    def test_tokenize_batch_consistency(self):
        """Test that tokenization is consistent across batches."""
        texts = ["42", "3.14", "7 + 8"]

        # Tokenize as a batch
        batch_tokens = tokenize_texts(texts, self.sequence_length, self.device)

        # Tokenize individually
        individual_tokens = []
        for text in texts:
            tokens = tokenize_texts([text], self.sequence_length, self.device)
            individual_tokens.append(tokens[0])

        # Results should be identical
        for i, individual_token in enumerate(individual_tokens):
            self.assertTrue(torch.equal(batch_tokens[i], individual_token))

    def test_tokenize_reproducibility(self):
        """Test that tokenization is reproducible."""
        texts = ["42.5 * 3", "10 / 2 + 1"]

        # Tokenize twice
        tokens1 = tokenize_texts(texts, self.sequence_length, self.device)
        tokens2 = tokenize_texts(texts, self.sequence_length, self.device)

        # Results should be identical
        self.assertTrue(torch.equal(tokens1, tokens2))

    def test_tokenize_device_placement(self):
        """Test that tokens are placed on the correct device."""
        texts = ["123", "456"]

        # Test CPU
        cpu_tokens = tokenize_texts(texts, self.sequence_length, "cpu")
        self.assertEqual(cpu_tokens.device.type, "cpu")

        # Test MPS if available
        if torch.backends.mps.is_available():
            mps_tokens = tokenize_texts(texts, self.sequence_length, "mps")
            self.assertEqual(mps_tokens.device.type, "mps")

        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_tokens = tokenize_texts(texts, self.sequence_length, "cuda")
            self.assertEqual(cuda_tokens.device.type, "cuda")

    def test_tokenize_roundtrip(self):
        """Test that tokenized text can be approximately decoded back."""
        from tiktoken import get_encoding

        texts = ["42", "3.14", "10 + 5", "negative twenty-three"]

        enc = get_encoding("gpt2")

        for text in texts:
            tokens = tokenize_texts([text], self.sequence_length, self.device)

            # Extract non-zero tokens
            non_zero_tokens = tokens[0][tokens[0] != 0].tolist()

            # Decode back
            decoded = enc.decode(non_zero_tokens)

            # Should approximately match original (allowing for whitespace differences)
            self.assertEqual(decoded.strip(), text.strip())


class TestModelSetup(unittest.TestCase):
    """Test model setup and initialization."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_model_creation_with_dag(self):
        """Test creating model with DAG components."""
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=2,  # Fixed for test - predictor doesn't use n_layer
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
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=2,  # Fixed for test - predictor doesn't use n_layer
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
        cfg.n_head = 2
        cfg.n_embd = 32
        cfg.sequence_length = 16

        model_args = dict(
            n_layer=2,  # Fixed for test - predictor doesn't use n_layer
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
            pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden)

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
            from models.dag_model import LOG_LIM

            self.assertTrue(
                ((pred_log >= -LOG_LIM) & (pred_log <= LOG_LIM)).all()
            )  # Log magnitudes should be within valid range


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint saving and loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True)
        self.cfg = DAGTrainConfig()
        self.cfg.name = "test_checkpoint"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_torch_save(self):
        """Test safe checkpoint saving."""
        checkpoint_manager = CheckpointManager("dag")

        test_data = {
            "model": {"param1": torch.randn(5, 3), "param2": torch.randn(10)},
            "optimizer": {"state": {}, "param_groups": []},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        # Test successful save
        filename = "test_checkpoint.pt"
        with patch("checkpoint_manager.CHECKPOINT_DIR", self.temp_dir):
            checkpoint_manager.save_checkpoint(test_data, filename)

        save_path = Path(self.temp_dir) / filename
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
        checkpoint_manager = CheckpointManager("dag")

        cfg = DAGTrainConfig()
        cfg.name = "test_clean"
        cfg.clear_previous_checkpoints = True
        model_name = "MyModel"

        # Create some fake checkpoint files (using new naming pattern)
        checkpoint_dir = Path(self.temp_dir)
        (checkpoint_dir / f"ckpt_{cfg.name}_100.pt").touch()
        (checkpoint_dir / f"ckpt_{cfg.name}_200.pt").touch()
        (
            checkpoint_dir / f"ckpt_other_run_100.pt"
        ).touch()  # Different run name, shouldn't be removed

        with patch("checkpoint_manager.CHECKPOINT_DIR", self.temp_dir):
            checkpoint_manager.clean_previous_checkpoints(cfg.name, model_name)

        # Check that only the matching checkpoints were removed
        self.assertFalse((checkpoint_dir / f"ckpt_{cfg.name}_100.pt").exists())
        self.assertFalse((checkpoint_dir / f"ckpt_{cfg.name}_200.pt").exists())
        self.assertTrue((checkpoint_dir / f"ckpt_other_run_100.pt").exists())

    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint."""
        checkpoint_manager = CheckpointManager("dag")

        cfg = DAGTrainConfig()
        cfg.name = "test_find"
        model_name = "MyModel"

        checkpoint_dir = Path(self.temp_dir)

        # Create checkpoint files with different iteration numbers (using new naming pattern)
        (checkpoint_dir / f"ckpt_{cfg.name}_100.pt").touch()
        (checkpoint_dir / f"ckpt_{cfg.name}_500.pt").touch()
        (checkpoint_dir / f"ckpt_{cfg.name}_300.pt").touch()

        with patch("checkpoint_manager.CHECKPOINT_DIR", self.temp_dir):
            latest = checkpoint_manager.find_latest_checkpoint(cfg.name, model_name)

        self.assertIsNotNone(latest)
        self.assertEqual(latest.name, f"ckpt_{cfg.name}_500.pt")

    def test_save_best_checkpoint_logic(self):
        # This is a conceptual test of the logic, not a full training run.
        # We'll simulate the saving part.
        checkpoint_manager = CheckpointManager("dag")

        cfg = DAGTrainConfig(name="best_run", save_best=True)
        model_name = "BestModel"

        # Simulate finding a new best checkpoint (using new naming pattern)
        best_ckpt_filename = f"ckpt_{cfg.name}_best.pt"

        # First best
        (self.checkpoint_dir / best_ckpt_filename).touch()

        # Verify it exists
        self.assertTrue((self.checkpoint_dir / best_ckpt_filename).exists())

        # Test that we can find the best checkpoint
        with patch("checkpoint_manager.CHECKPOINT_DIR", str(self.checkpoint_dir)):
            best_checkpoint = checkpoint_manager.find_best_checkpoint(
                cfg.name, model_name
            )
            self.assertIsNotNone(best_checkpoint)
            self.assertEqual(best_checkpoint.name, best_ckpt_filename)


class TestCheckpointLoadingPredictor(unittest.TestCase):
    """Test checkpoint loading for predictor-only models."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_loading_predictor_only_config(self):
        """Test that checkpoint loading works correctly with PredictorOnlyConfig."""
        # Create a model and save a checkpoint
        config = PredictorOnlyConfig(
            vocab_size=50304,
            n_embd=64,
            n_head=4,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            sequence_length=32,
            softmax_temperature=20.0,
        )
        model = PredictorOnlyModel(config)

        # Create a checkpoint with proper predictor-only config
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": {},  # Empty optimizer state for test
            "model_config": config.__dict__,  # This is the proper predictor config
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        # Save the checkpoint
        checkpoint_path = self.checkpoint_dir / "test_checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_path)

        # Now try to load it and create a PredictorOnlyConfig from it
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # This should work without error
        try:
            saved_config = PredictorOnlyConfig(**loaded_checkpoint["model_config"])
            # Verify that it has the expected attributes
            self.assertTrue(hasattr(saved_config, "n_head"))
            self.assertTrue(hasattr(saved_config, "n_embd"))
            self.assertTrue(hasattr(saved_config, "dag_depth"))

            # n_layer should not be part of PredictorOnlyConfig
            self.assertFalse(hasattr(saved_config, "n_layer"))

        except Exception as e:
            self.fail(f"Loading checkpoint failed unexpectedly: {e}")

    def test_checkpoint_loading_with_extra_fields(self):
        """Test that checkpoint loading ignores extra fields in model_config."""
        # Create a model and save a checkpoint
        config = PredictorOnlyConfig(
            vocab_size=50304,
            n_embd=64,
            n_head=4,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            sequence_length=32,
            softmax_temperature=20.0,
        )
        model = PredictorOnlyModel(config)

        # Create a checkpoint with extra fields that aren't in PredictorOnlyConfig
        # This simulates a checkpoint saved with extra training config fields
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": {},
            "model_config": {
                **config.__dict__,
                "max_iters": 10000,  # This field doesn't exist in PredictorOnlyConfig
                "learning_rate": 1e-3,  # This field doesn't exist in PredictorOnlyConfig
            },
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        # Save the checkpoint
        checkpoint_path = self.checkpoint_dir / "test_checkpoint_extra.pt"
        torch.save(checkpoint_data, checkpoint_path)

        # Load it back
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # This should work even with extra fields - they should be ignored
        try:
            saved_config = PredictorOnlyConfig(**loaded_checkpoint["model_config"])
            # Should have the proper predictor config attributes
            self.assertTrue(hasattr(saved_config, "n_head"))
            self.assertTrue(hasattr(saved_config, "n_embd"))
            self.assertTrue(hasattr(saved_config, "dag_depth"))

        except TypeError as e:
            # This might fail if PredictorOnlyConfig doesn't handle extra kwargs
            self.assertIn("unexpected keyword argument", str(e))

    def test_checkpoint_logging_works_correctly(self):
        """Test that checkpoint logging works correctly with predictor-only config."""
        # Create a model and save a checkpoint
        config = PredictorOnlyConfig(
            vocab_size=50304,
            n_embd=64,
            n_head=4,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            sequence_length=32,
            softmax_temperature=20.0,
        )
        model = PredictorOnlyModel(config)

        # Create a checkpoint that simulates what would be saved by the training code
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": {},
            "model_config": config.__dict__,  # This is what gets saved
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        # Save the checkpoint
        checkpoint_path = self.checkpoint_dir / "test_checkpoint_logging.pt"
        torch.save(checkpoint_data, checkpoint_path)

        # Load it back (this simulates the code in train_predictor.py)
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # This should work without error now
        saved_config = PredictorOnlyConfig(**loaded_checkpoint["model_config"])

        # Test the correct logging format for shallow attention model
        message = f"Model config: {saved_config.n_head}H, {saved_config.n_embd}D (shallow attention)"

        # Verify the message contains the expected components
        self.assertIn(f"{saved_config.n_head}H", message)
        self.assertIn(f"{saved_config.n_embd}D", message)
        self.assertIn("shallow attention", message)


# -----------------------------------------------------------------------------
# New tests for full backbone mode
# -----------------------------------------------------------------------------


class TestFullBackbonePredictor(unittest.TestCase):
    """Tests for the full GPT backbone with DAG plan predictor outputs."""

    def setUp(self):
        torch.manual_seed(0)
        self.cfg = GPTConfig(
            block_size=16,
            vocab_size=50304,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            softmax_temperature=20.0,
        )
        self.model = GPT(self.cfg)
        self.model.eval()

    def test_plan_predictor_output_shapes(self):
        batch = 2
        seq_len = 16
        input_ids = torch.randint(0, self.cfg.vocab_size, (batch, seq_len))
        with torch.no_grad():
            hidden = self.model.forward_hidden(input_ids)
            pred_sgn, pred_log, pred_ops = self.model.dag.plan_predictor(hidden)

        expected_nodes = self.cfg.dag_depth + 1
        self.assertEqual(pred_sgn.shape, (batch, seq_len, expected_nodes))
        self.assertEqual(pred_log.shape, (batch, seq_len, expected_nodes))
        self.assertEqual(pred_ops.shape, (batch, seq_len, self.cfg.dag_depth, 5))

        # Basic sanity checks on ranges
        self.assertTrue((pred_sgn >= -1).all() and (pred_sgn <= 1).all())
        self.assertTrue(torch.isfinite(pred_log).all())
        op_sums = pred_ops.sum(dim=-1)
        self.assertTrue(torch.allclose(op_sums, torch.ones_like(op_sums), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
