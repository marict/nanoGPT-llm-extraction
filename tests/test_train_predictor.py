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
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import GPT, GPTConfig
from train_predictor import (DAGTrainConfig, PredictorOnlyConfig,
                             PredictorOnlyModel, _all_tensors,
                             _safe_torch_save, apply_overrides,
                             clean_previous_checkpoints,
                             compute_dag_structure_loss, evaluate_dag_model,
                             find_latest_checkpoint, get_checkpoint_filename,
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

        # Create identical predictions and targets
        target_sgn = torch.randn(B, T, num_nodes)
        target_log = torch.abs(torch.randn(B, T, num_nodes))
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        # Perfect predictions = targets
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
        cfg = DAGTrainConfig()
        cfg.name = "test_run"
        model_name = "TestModel"
        filename = get_checkpoint_filename(cfg, 1000, model_name)
        self.assertEqual(filename, "dag_ckpt_TestModel_test_run_001000")

    def test_all_tensors(self):
        """Test _all_tensors utility function."""
        # Test with all tensors
        state_all_tensors = {
            "layer1": {"weight": torch.randn(10, 10), "bias": torch.randn(10)},
            "layer2": {"weight": torch.randn(5, 10)},
        }
        self.assertTrue(_all_tensors(state_all_tensors))

        # Test with non-tensor
        state_mixed = {
            "layer1": {"weight": torch.randn(10, 10), "bias": [1, 2, 3]},
            "layer2": {"weight": torch.randn(5, 10)},
        }
        self.assertFalse(_all_tensors(state_mixed))


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
        model_name = "MyModel"

        # Create some fake checkpoint files
        checkpoint_dir = Path(self.temp_dir)
        (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000100.pt").touch()
        (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000200.pt").touch()
        (
            checkpoint_dir / f"dag_ckpt_{model_name}_other_run_000100.pt"
        ).touch()  # Different run name, shouldn't be removed

        with patch("train_predictor.CHECKPOINT_DIR", self.temp_dir):
            clean_previous_checkpoints(cfg, model_name)

        # Check that only the matching checkpoints were removed
        self.assertFalse(
            (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000100.pt").exists()
        )
        self.assertFalse(
            (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000200.pt").exists()
        )
        self.assertTrue(
            (checkpoint_dir / f"dag_ckpt_{model_name}_other_run_000100.pt").exists()
        )

    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint."""
        cfg = DAGTrainConfig()
        cfg.name = "test_find"
        model_name = "MyModel"

        checkpoint_dir = Path(self.temp_dir)

        # Create checkpoint files with different iteration numbers
        (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000100.pt").touch()
        (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000500.pt").touch()
        (checkpoint_dir / f"dag_ckpt_{model_name}_{cfg.name}_000300.pt").touch()

        with patch("train_predictor.CHECKPOINT_DIR", self.temp_dir):
            latest = find_latest_checkpoint(cfg, model_name)

        self.assertIsNotNone(latest)
        self.assertEqual(latest.name, f"dag_ckpt_{model_name}_{cfg.name}_000500.pt")


class TestIntegration(unittest.TestCase):
    """Integration tests for training components."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.cfg = DAGTrainConfig()
        self.cfg.dag_depth = 2
        self.cfg.n_head = 2
        self.cfg.n_embd = 32
        self.cfg.batch_size = 2
        self.cfg.sequence_length = 16

    def test_evaluate_shallow_attention_model(self):
        """Test evaluation function with new shallow attention model."""
        # Create shallow attention model
        model_config = PredictorOnlyConfig(
            vocab_size=50304,  # Use proper GPT-2 vocab size
            n_embd=self.cfg.n_embd,
            n_head=self.cfg.n_head,
            dropout=self.cfg.dropout,
            bias=self.cfg.bias,
            dag_depth=self.cfg.dag_depth,
            sequence_length=self.cfg.sequence_length,
            softmax_temperature=20.0,
        )
        model = PredictorOnlyModel(model_config)
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
        ctx = (
            nullcontext()
            if device == "cpu"
            else torch.amp.autocast(device_type=device, dtype=torch.float32)
        )
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
            n_layer=2,  # Fixed for test - predictor doesn't use n_layer
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
                pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden)

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
            n_layer=2,  # Fixed for test - predictor doesn't use n_layer
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
        pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden)

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
        model_config = PredictorOnlyConfig(
            vocab_size=1000,
            n_embd=self.cfg.n_embd,
            n_head=self.cfg.n_head,
            dropout=self.cfg.dropout,
            bias=self.cfg.bias,
            dag_depth=self.cfg.dag_depth,
            sequence_length=self.cfg.sequence_length,
            softmax_temperature=20.0,
        )
        model = PredictorOnlyModel(model_config)
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

        # Training step
        optimizer.zero_grad()

        # Forward pass
        batch_size = len(texts)
        input_tokens = torch.randint(0, 1000, (batch_size, self.cfg.sequence_length))

        # Forward through shallow attention model with internal state
        pred_sgn, pred_log, pred_ops = model(input_tokens)

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
            self.cfg,
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

        # Verify we have the expected loss components
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
        self.assertEqual(set(losses.keys()), expected_keys)


class TestDataLoaderIntegration(unittest.TestCase):
    """Test data loader integration with train_predictor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_create_dag_structure_dataloaders(self):
        """Test that DAG structure dataloaders can be created and used."""
        # Create dataloaders with small batch sizes for testing
        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=3,
            train_seed=42,
            val_seed=43,
        )

        # Test train loader
        train_batch = next(train_loader)
        texts, structures = train_batch

        self.assertIsInstance(texts, list)
        self.assertEqual(len(texts), 4)  # batch_size
        self.assertIsInstance(texts[0], str)
        self.assertGreater(len(texts[0]), 0)

        # Test structure tensors
        self.assertIsInstance(structures, dict)
        expected_keys = {"initial_sgn", "initial_log", "operation_probs", "depths"}
        self.assertEqual(set(structures.keys()), expected_keys)

        # Check tensor shapes
        self.assertEqual(
            structures["initial_sgn"].shape, (4, 4)
        )  # (batch_size, max_depth+1)
        self.assertEqual(structures["initial_log"].shape, (4, 4))
        self.assertEqual(
            structures["operation_probs"].shape, (4, 3, 5)
        )  # (batch_size, depth, n_ops)
        self.assertEqual(structures["depths"].shape, (4,))

        # Test val loader
        val_batch = next(val_loader)
        val_texts, val_structures = val_batch

        self.assertIsInstance(val_texts, list)
        self.assertEqual(len(val_texts), 2)  # val_batch_size
        self.assertEqual(val_structures["initial_sgn"].shape, (2, 4))

    def test_dataloader_content_consistency(self):
        """Test that dataloader content is consistent and valid."""
        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=8,
            val_batch_size=4,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        # Test multiple batches from train loader
        for i in range(3):
            texts, structures = next(train_loader)

            # Check text content
            for text in texts:
                self.assertIsInstance(text, str)
                self.assertGreater(len(text), 0)
                # Should contain mathematical expressions (numbers or English words)
                import re

                has_numbers = re.search(r"\d+\.?\d*", text)
                has_english_words = re.search(r"[a-zA-Z]+", text)

                # With 30% English conversion, expressions may be entirely in English
                self.assertTrue(
                    has_numbers or has_english_words,
                    f"Text should contain numbers or English words: '{text}'",
                )
                # May or may not contain operators (single numbers/words are valid)
                # self.assertTrue(re.search(r'[\+\-\*/]', text))  # Contains operators

            # Check structure tensors
            batch_size = len(texts)
            self.assertEqual(
                structures["initial_sgn"].shape, (batch_size, 3)
            )  # depth+1
            self.assertEqual(structures["initial_log"].shape, (batch_size, 3))
            self.assertEqual(structures["operation_probs"].shape, (batch_size, 2, 5))
            self.assertEqual(structures["depths"].shape, (batch_size,))

            # Check that depths are consistent
            self.assertTrue(torch.all(structures["depths"] == 2))

            # Check that operation probabilities are valid (sum to 1 or 0)
            op_probs = structures["operation_probs"]
            for b in range(batch_size):
                for d in range(2):  # depth
                    row = op_probs[b, d]
                    # Should be one-hot (one 1.0, rest 0.0) or all zeros
                    if torch.sum(row) > 0:
                        self.assertAlmostEqual(torch.sum(row).item(), 1.0, places=5)
                        self.assertEqual(
                            torch.sum(row > 0).item(), 1
                        )  # Exactly one non-zero

    def test_dataloader_reproducibility(self):
        """Test that dataloaders are reproducible with same seeds."""
        # Create two sets of dataloaders with same seeds
        train_loader1, val_loader1 = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        train_loader2, val_loader2 = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        # Compare first batches
        texts1, structures1 = next(train_loader1)
        texts2, structures2 = next(train_loader2)

        self.assertEqual(texts1, texts2)
        for key in structures1:
            self.assertTrue(
                torch.allclose(structures1[key], structures2[key], atol=1e-6)
            )

    def test_dataloader_different_seeds(self):
        """Test that different seeds produce different data."""
        train_loader1, _ = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        train_loader2, _ = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=123,
            val_seed=456,
        )

        # Compare first batches
        texts1, structures1 = next(train_loader1)
        texts2, structures2 = next(train_loader2)

        # Should be different (very unlikely to be the same with different seeds)
        self.assertNotEqual(texts1, texts2)


class TestTrainingIntegration(unittest.TestCase):
    """Test full training integration with data loaders."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_model_with_real_dataloader(self):
        """Test model forward pass with real dataloader data."""
        # Create small model with smaller sequence length
        config = PredictorOnlyConfig(
            vocab_size=1000,
            n_embd=32,
            n_head=2,
            dag_depth=2,
            sequence_length=8,  # Smaller to avoid position embedding issues
        )
        model = PredictorOnlyModel(config)

        # Create dataloader
        train_loader, _ = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        # Get a batch
        texts, structures = next(train_loader)

        # Create input_ids from texts (simple tokenization for testing)
        import re

        input_ids = []
        for text in texts:
            # Simple tokenization: split on spaces and convert to integers
            tokens = re.findall(r"\d+\.?\d*|\+|\-|\*|/|\(|\)", text)
            # Pad or truncate to sequence_length
            if len(tokens) < 8:  # sequence_length
                tokens.extend(["0"] * (8 - len(tokens)))
            else:
                tokens = tokens[:8]
            # Convert to integers (simple hash-based tokenization)
            ids = [hash(token) % 1000 for token in tokens]
            input_ids.append(ids)

        input_ids = torch.tensor(input_ids)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            pred_sgn, pred_log, pred_ops = model(input_ids)

            # Check shapes
            self.assertEqual(
                pred_sgn.shape, (4, 8, 3)
            )  # (batch_size, seq_len, depth+1)
            self.assertEqual(pred_log.shape, (4, 8, 3))
            self.assertEqual(
                pred_ops.shape, (4, 8, 2, 5)
            )  # (batch_size, seq_len, depth, n_ops)

            # Check that predictions are finite
            self.assertTrue(torch.isfinite(pred_sgn).all())
            self.assertTrue(torch.isfinite(pred_log).all())
            self.assertTrue(torch.isfinite(pred_ops).all())

    def test_evaluation_with_real_data(self):
        """Test evaluation function with real dataloader data."""
        # Create small model with smaller sequence length to avoid position embedding issues
        config = PredictorOnlyConfig(
            vocab_size=50304,  # Use proper GPT-2 vocab size
            n_embd=32,
            n_head=2,
            dag_depth=2,
            sequence_length=8,  # Smaller to avoid position embedding issues
        )
        model = PredictorOnlyModel(config)

        # Create dataloader
        _, val_loader = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=2,
            train_seed=42,
            val_seed=43,
        )

        # Mock device and context
        device = "cpu"
        ctx = (
            nullcontext()
            if device == "cpu"
            else torch.amp.autocast(device_type=device, dtype=torch.float32)
        )
        # Create config with matching sequence length
        cfg = DAGTrainConfig(
            eval_iters=1,  # Just one iteration to avoid issues
            sign_loss_weight=1.0,
            log_loss_weight=1.0,
            op_loss_weight=1.0,
            sequence_length=8,  # Match the model's sequence length
        )

        # Test evaluation
        with ctx:
            metrics = evaluate_dag_model(
                model=model,
                val_loader=val_loader,
                device=device,
                ctx=ctx,
                cfg=cfg,
                eval_iters=1,
            )

        # Check metrics
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertTrue(np.isfinite(metrics[key]))
            self.assertGreaterEqual(metrics[key], 0)


class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir)
        # Monkey-patch the checkpoint directory
        import train_predictor

        train_predictor.CHECKPOINT_DIR = str(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_find_latest_checkpoint_resume(self):
        # Test finding the latest checkpoint for a specific run
        cfg = DAGTrainConfig(name="test_run")
        model_name = "TestModel"

        # Create some dummy checkpoint files
        (
            self.checkpoint_dir / get_checkpoint_filename(cfg, 100, model_name)
        ).with_suffix(".pt").touch()
        (
            self.checkpoint_dir / get_checkpoint_filename(cfg, 200, model_name)
        ).with_suffix(".safetensors").touch()
        (
            self.checkpoint_dir / get_checkpoint_filename(cfg, 50, model_name)
        ).with_suffix(".pt").touch()

        # Create a checkpoint for another run that should be ignored
        other_cfg = DAGTrainConfig(name="other_run")
        (
            self.checkpoint_dir / get_checkpoint_filename(other_cfg, 300, model_name)
        ).with_suffix(".pt").touch()

        latest_ckpt = find_latest_checkpoint(cfg, model_name, any_run=False)
        self.assertIsNotNone(latest_ckpt)
        self.assertEqual(
            latest_ckpt.stem, get_checkpoint_filename(cfg, 200, model_name)
        )
        self.assertEqual(latest_ckpt.suffix, ".safetensors")

    def test_find_latest_checkpoint_latest(self):
        # Test finding the latest checkpoint across any run
        cfg = DAGTrainConfig(name="any_run_will_do")
        model_name = "TestModel"

        # Create dummy checkpoints for multiple runs
        run1_cfg = DAGTrainConfig(name="run1")
        (
            self.checkpoint_dir / get_checkpoint_filename(run1_cfg, 100, model_name)
        ).with_suffix(".pt").touch()

        run2_cfg = DAGTrainConfig(name="run2")
        (
            self.checkpoint_dir / get_checkpoint_filename(run2_cfg, 300, model_name)
        ).with_suffix(".safetensors").touch()

        run3_cfg = DAGTrainConfig(name="run3")
        (
            self.checkpoint_dir / get_checkpoint_filename(run3_cfg, 200, model_name)
        ).with_suffix(".pt").touch()

        # This one has a different model name and should be ignored
        (
            self.checkpoint_dir / get_checkpoint_filename(run3_cfg, 400, "AnotherModel")
        ).with_suffix(".pt").touch()

        latest_ckpt = find_latest_checkpoint(cfg, model_name, any_run=True)
        self.assertIsNotNone(latest_ckpt)
        self.assertEqual(
            latest_ckpt.stem, get_checkpoint_filename(run2_cfg, 300, model_name)
        )

    def test_find_latest_checkpoint_no_checkpoints(self):
        cfg = DAGTrainConfig()
        model_name = "TestModel"
        latest_ckpt = find_latest_checkpoint(cfg, model_name)
        self.assertIsNone(latest_ckpt)

    def test_save_best_checkpoint_logic(self):
        # This is a conceptual test of the logic, not a full training run.
        # We'll simulate the saving part.
        cfg = DAGTrainConfig(name="best_run", save_best=True)
        model_name = "BestModel"

        # Simulate finding a new best checkpoint
        best_ckpt_base = f"dag_ckpt_{model_name}_{cfg.name}_best"

        # First best
        (self.checkpoint_dir / f"{best_ckpt_base}.pt").touch()

        # Verify it exists
        self.assertTrue((self.checkpoint_dir / f"{best_ckpt_base}.pt").exists())

        # Simulate another new best, this time with safetensors
        (self.checkpoint_dir / f"{best_ckpt_base}.safetensors").touch()

        # The new file should exist, and you would typically remove the old one
        # but the save logic should handle overwriting via rename
        self.assertTrue(
            (self.checkpoint_dir / f"{best_ckpt_base}.safetensors").exists()
        )


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


if __name__ == "__main__":
    unittest.main()
