#!/usr/bin/env python
"""
test_train_dag.py
Tests for the DAG predictor training script.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dag_model import GPT, GPTConfig
from data.dagset.streaming import create_dag_structure_dataloaders
from train_dag import (DAGTrainConfig, _all_tensors, _safe_torch_save,
                       apply_overrides, clean_previous_checkpoints,
                       compute_dag_structure_loss, evaluate_dag_model,
                       find_latest_checkpoint, get_checkpoint_filename, get_lr,
                       load_config_file, update_config)


class TestDAGTrainConfig(unittest.TestCase):
    """Test DAG training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = DAGTrainConfig()

        self.assertEqual(cfg.name, "dag_pretrain")
        self.assertEqual(cfg.eval_interval, 100)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.dag_depth, 4)
        self.assertEqual(cfg.learning_rate, 3e-4)
        self.assertEqual(cfg.max_iters, 10000)
        self.assertEqual(cfg.sign_loss_weight, 1.0)
        self.assertEqual(cfg.log_loss_weight, 1.0)
        self.assertEqual(cfg.op_loss_weight, 1.0)

    def test_config_attributes(self):
        """Test that config has all required attributes."""
        cfg = DAGTrainConfig()

        required_attrs = [
            "eval_interval",
            "log_interval",
            "eval_iters",
            "eval_only",
            "always_save_checkpoint",
            "clear_previous_checkpoints",
            "init_from",
            "name",
            "max_dag_depth",
            "min_dag_depth",
            "train_examples_per_batch",
            "val_examples_per_batch",
            "gradient_accumulation_steps",
            "batch_size",
            "sequence_length",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "bias",
            "dag_depth",
            "learning_rate",
            "max_iters",
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
            "sign_loss_weight",
            "log_loss_weight",
            "op_loss_weight",
            "train_seed",
            "val_seed",
        ]

        for attr in required_attrs:
            self.assertTrue(hasattr(cfg, attr), f"Config missing attribute: {attr}")


class TestConfigManagement(unittest.TestCase):
    """Test configuration loading and management functions."""

    def test_load_config_file(self):
        """Test loading configuration from Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
# Test config
name = "test_project"
learning_rate = 1e-3
batch_size = 64
dag_depth = 2
"""
            )
            f.flush()

            try:
                config_data = load_config_file(f.name)

                self.assertEqual(config_data["name"], "test_project")
                self.assertEqual(config_data["learning_rate"], 1e-3)
                self.assertEqual(config_data["batch_size"], 64)
                self.assertEqual(config_data["dag_depth"], 2)

                # Should not include __builtins__ etc.
                self.assertNotIn("__builtins__", config_data)
            finally:
                os.unlink(f.name)

    def test_update_config(self):
        """Test updating configuration with dictionary."""
        cfg = DAGTrainConfig()
        original_lr = cfg.learning_rate

        data = {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "max_iters": 1000,
            "nonexistent_attr": "ignored",  # Should be ignored
        }

        update_config(cfg, data)

        self.assertEqual(cfg.learning_rate, 1e-4)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.max_iters, 1000)
        self.assertFalse(hasattr(cfg, "nonexistent_attr"))

    def test_apply_overrides(self):
        """Test applying command line overrides."""
        cfg = DAGTrainConfig()

        overrides = [
            "--learning_rate=2e-4",
            "--batch_size=128",
            "--dag_depth=6",
            "--dropout=0.2",
            "--compile=False",
            "invalid_override",  # Should be ignored
            "--nonexistent=value",  # Should be ignored
        ]

        apply_overrides(cfg, overrides)

        self.assertEqual(cfg.learning_rate, 2e-4)
        self.assertEqual(cfg.batch_size, 128)
        self.assertEqual(cfg.dag_depth, 6)
        self.assertEqual(cfg.dropout, 0.2)
        self.assertEqual(cfg.compile, False)


class TestLossFunctions(unittest.TestCase):
    """Test DAG structure loss computation."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.cfg = DAGTrainConfig()
        self.batch_size = 2
        self.seq_len = 1
        self.num_nodes = 5
        self.depth = 4
        self.n_ops = 5

    def test_compute_dag_structure_loss_shapes(self):
        """Test loss computation with correct tensor shapes."""
        # Create test tensors
        pred_sgn = torch.randn(self.batch_size, self.seq_len, self.num_nodes)
        pred_log = torch.abs(torch.randn(self.batch_size, self.seq_len, self.num_nodes))
        pred_ops = torch.softmax(
            torch.randn(self.batch_size, self.seq_len, self.depth, self.n_ops), dim=-1
        )

        target_sgn = torch.randn(self.batch_size, self.seq_len, self.num_nodes)
        target_log = torch.abs(
            torch.randn(self.batch_size, self.seq_len, self.num_nodes)
        )

        # Create one-hot operation targets
        target_ops = torch.zeros(self.batch_size, self.seq_len, self.depth, self.n_ops)
        for b in range(self.batch_size):
            for t in range(self.seq_len):
                for d in range(self.depth):
                    op_idx = torch.randint(0, self.n_ops, (1,)).item()
                    target_ops[b, t, d, op_idx] = 1.0

        target_depths = torch.tensor([self.depth] * self.batch_size)

        # Compute losses
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

        # Check that all losses are returned and are scalars
        expected_keys = {"total_loss", "sign_loss", "log_loss", "op_loss"}
        self.assertEqual(set(losses.keys()), expected_keys)

        for key, loss in losses.items():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))  # Scalar
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss.item(), 0.0)

    def test_compute_dag_structure_loss_weights(self):
        """Test that loss weights are applied correctly."""
        cfg = DAGTrainConfig()
        cfg.sign_loss_weight = 2.0
        cfg.log_loss_weight = 1.0
        cfg.op_loss_weight = 3.0

        # Create simple test tensors
        pred_sgn = torch.ones(1, 1, 3)
        pred_log = torch.ones(1, 1, 3)
        pred_ops = torch.ones(1, 1, 2, 5) / 5  # Uniform distribution

        target_sgn = torch.zeros(1, 1, 3)
        target_log = torch.zeros(1, 1, 3)
        target_ops = torch.zeros(1, 1, 2, 5)
        target_ops[0, 0, :, 0] = 1.0  # First operation for all steps

        target_depths = torch.tensor([2])

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_log,
            pred_ops,
            target_sgn,
            target_log,
            target_ops,
            target_depths,
            cfg,
        )

        # Verify that total loss incorporates weights
        expected_total = (
            cfg.sign_loss_weight * losses["sign_loss"]
            + cfg.log_loss_weight * losses["log_loss"]
            + cfg.op_loss_weight * losses["op_loss"]
        )

        self.assertAlmostEqual(
            losses["total_loss"].item(), expected_total.item(), places=5
        )

    def test_compute_dag_structure_loss_perfect_prediction(self):
        """Test loss computation with perfect predictions."""
        # Create identical predictions and targets
        target_sgn = torch.randn(1, 1, 3)
        target_log = torch.abs(torch.randn(1, 1, 3))
        target_ops = torch.zeros(1, 1, 2, 5)
        target_ops[0, 0, 0, 1] = 1.0  # Operation 1 for step 0
        target_ops[0, 0, 1, 3] = 1.0  # Operation 3 for step 1

        target_depths = torch.tensor([2])

        # Use same tensors for predictions but ensure they're probability distributions
        pred_sgn = target_sgn.clone()
        pred_log = target_log.clone()

        # For operations, create predictions that exactly match targets
        # Start with a small uniform distribution then set the correct class to high probability
        pred_ops = torch.full_like(target_ops, 0.001 / target_ops.size(-1))

        # Set high probability for the correct operations
        # target_ops[0, 0, 0, 1] = 1.0  means operation 1 for step 0
        # target_ops[0, 0, 1, 3] = 1.0  means operation 3 for step 1
        pred_ops[0, 0, 0, 1] = 0.999  # Set high prob for operation 1, step 0
        pred_ops[0, 0, 1, 3] = 0.999  # Set high prob for operation 3, step 1

        # Normalize to ensure they're valid probability distributions
        pred_ops = pred_ops / pred_ops.sum(dim=-1, keepdim=True)

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

        # All losses should be very small for near-perfect predictions
        self.assertLess(losses["sign_loss"].item(), 1e-6)
        self.assertLess(losses["log_loss"].item(), 1e-6)
        self.assertLess(losses["op_loss"].item(), 0.01)  # -log(0.999) ≈ 0.001
        self.assertLess(losses["total_loss"].item(), 0.01)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_lr_warmup(self):
        """Test learning rate during warmup phase."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-3
        cfg.warmup_iters = 100
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        # Test warmup phase
        lr_start = get_lr(0, cfg=cfg)
        lr_mid = get_lr(50, cfg=cfg)
        lr_end = get_lr(100, cfg=cfg)

        self.assertAlmostEqual(lr_start, 0.0, places=6)
        self.assertAlmostEqual(lr_mid, cfg.learning_rate * 0.5, places=6)
        self.assertAlmostEqual(lr_end, cfg.learning_rate, places=6)

    def test_get_lr_decay(self):
        """Test learning rate during decay phase."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-3
        cfg.warmup_iters = 100
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        # Test decay phase
        lr_start_decay = get_lr(100, cfg=cfg)
        lr_mid_decay = get_lr(550, cfg=cfg)
        lr_end_decay = get_lr(1000, cfg=cfg)
        lr_after_decay = get_lr(1500, cfg=cfg)

        self.assertAlmostEqual(lr_start_decay, cfg.learning_rate, places=6)
        self.assertGreater(lr_mid_decay, cfg.min_lr)
        self.assertLess(lr_mid_decay, cfg.learning_rate)
        self.assertAlmostEqual(lr_end_decay, cfg.min_lr, places=6)
        self.assertAlmostEqual(lr_after_decay, cfg.min_lr, places=6)

    def test_get_checkpoint_filename(self):
        """Test checkpoint filename generation."""
        cfg = DAGTrainConfig()
        cfg.name = "test_run"

        filename = get_checkpoint_filename(cfg, 1234)
        expected = "dag_ckpt_test_run_001234.pt"

        self.assertEqual(filename, expected)

    def test_all_tensors(self):
        """Test _all_tensors utility function."""
        # All tensors
        state_all_tensors = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2": {"weight": torch.randn(5, 3), "bias": torch.randn(5)},
        }
        self.assertTrue(_all_tensors(state_all_tensors))

        # Mixed types
        state_mixed = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": [1, 2, 3],  # Not a tensor
            "config": {"param": 42},
        }
        self.assertFalse(_all_tensors(state_mixed))


class TestModelSetup(unittest.TestCase):
    """Test model setup and parameter freezing."""

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

        # Verify model has DAG components
        self.assertTrue(hasattr(model, "dag"))
        self.assertIsNotNone(model.dag)
        self.assertTrue(hasattr(model.dag, "plan_predictor"))

    def test_parameter_freezing(self):
        """Test that non-DAG parameters are properly frozen."""
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.n_layer = 2
        cfg.n_head = 2
        cfg.n_embd = 32

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

        # Freeze non-DAG parameters (simulate train_dag.py behavior)
        for name, param in model.named_parameters():
            if "dag.plan_predictor" not in name:
                param.requires_grad = False

        # Count trainable vs frozen parameters
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)

        self.assertGreater(trainable_params, 0)  # Should have some trainable params
        self.assertGreater(frozen_params, 0)  # Should have some frozen params

        # Verify DAG predictor parameters are trainable
        dag_trainable = sum(
            1
            for n, p in model.named_parameters()
            if "dag.plan_predictor" in n and p.requires_grad
        )
        self.assertGreater(dag_trainable, 0)

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

        with patch("train_dag.CHECKPOINT_DIR", self.temp_dir):
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

        with patch("train_dag.CHECKPOINT_DIR", self.temp_dir):
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

    def test_evaluate_dag_model_integration(self):
        """Test evaluation function integration."""
        # Create model
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
        from data.dagset.streaming import create_dag_structure_dataloaders

        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=self.cfg.batch_size,
            val_batch_size=self.cfg.batch_size,
            max_depth=self.cfg.dag_depth,
            min_depth=1,
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
        """Test a single training step integration."""
        # Create model
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
            min_depth=1,
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

        # Average over sequence dimension
        pred_sgn_avg = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
        pred_log_avg = pred_log.mean(dim=1)  # (B, num_nodes_pred)
        pred_ops_avg = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

        # Ensure target and prediction tensors have compatible shapes
        target_nodes = target_sgn.size(1)
        pred_nodes = pred_sgn_avg.size(1)
        target_depth = target_ops.size(1)
        pred_depth = pred_ops_avg.size(1)

        # Resize predictions to match targets if needed
        if pred_nodes != target_nodes:
            if pred_nodes > target_nodes:
                # Truncate predictions
                pred_sgn_avg = pred_sgn_avg[:, :target_nodes]
                pred_log_avg = pred_log_avg[:, :target_nodes]
            else:
                # Pad predictions with zeros
                import torch.nn.functional as F

                pad_nodes = target_nodes - pred_nodes
                pred_sgn_avg = F.pad(pred_sgn_avg, (0, pad_nodes))
                pred_log_avg = F.pad(pred_log_avg, (0, pad_nodes))

        if pred_depth != target_depth:
            if pred_depth > target_depth:
                # Truncate predictions
                pred_ops_avg = pred_ops_avg[:, :target_depth]
            else:
                # Pad predictions with zeros
                import torch.nn.functional as F

                pad_depth = target_depth - pred_depth
                pred_ops_avg = F.pad(pred_ops_avg, (0, 0, 0, pad_depth))

        # Add sequence dimension
        pred_sgn_seq = pred_sgn_avg.unsqueeze(1)
        pred_log_seq = pred_log_avg.unsqueeze(1)
        pred_ops_seq = pred_ops_avg.unsqueeze(1)

        target_sgn_seq = target_sgn.unsqueeze(1)
        target_log_seq = target_log.unsqueeze(1)
        target_ops_seq = target_ops.unsqueeze(1)

        # Compute loss
        losses = compute_dag_structure_loss(
            pred_sgn_seq,
            pred_log_seq,
            pred_ops_seq,
            target_sgn_seq,
            target_log_seq,
            target_ops_seq,
            target_depths,
            self.cfg,
        )

        loss = losses["total_loss"]

        # Backward pass
        loss.backward()

        # Verify gradients exist for DAG parameters
        dag_params_with_grad = 0
        for name, param in model.named_parameters():
            if "dag.plan_predictor" in name and param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
                )
                dag_params_with_grad += 1

        self.assertGreater(dag_params_with_grad, 0)

        # Optimization step
        optimizer.step()

        # Verify loss is reasonable
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
