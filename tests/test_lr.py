"""
Comprehensive tests for learning rate functionality.

This test suite verifies that the learning rate implementations work correctly,
including cosine decay, warmup, and cyclical modulation. It also tests edge
cases and mathematical correctness.
"""

import math
import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from training_utils import BaseConfig, get_lr


@dataclass
class LRTestConfig(BaseConfig):
    name: str = "test"
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    max_iters: int = 1000
    use_cyclical_lr: bool = False
    cyclical_lr_period: int = 200
    cyclical_lr_amplitude: float = 0.1


@dataclass
class CyclicalLRTestConfig(BaseConfig):
    name: str = "cyclical_test"
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    max_iters: int = 1000
    use_cyclical_lr: bool = True
    cyclical_lr_period: int = 200
    cyclical_lr_amplitude: float = 0.1


@dataclass
class _LRLoggingTestConfig(BaseConfig):
    """Minimal configuration for LR schedule testing."""

    name: str = "lr_logging_test"
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_iters: int = 5
    lr_decay_iters: int = 20
    max_iters: int = 20


class TestLRScheduler(unittest.TestCase):
    def test_cosine_schedule(self):
        """Test the cosine decay learning rate schedule."""
        cfg = LRTestConfig()

        # Test warmup phase
        self.assertAlmostEqual(get_lr(0, cfg=cfg), cfg.learning_rate / cfg.warmup_iters)
        self.assertAlmostEqual(get_lr(49, cfg=cfg), cfg.learning_rate * 0.5)
        self.assertAlmostEqual(get_lr(99, cfg=cfg), cfg.learning_rate)

        # Test decay phase
        mid_decay_it = cfg.warmup_iters + (cfg.lr_decay_iters - cfg.warmup_iters) // 2
        decay_ratio = (mid_decay_it - cfg.warmup_iters) / (
            cfg.lr_decay_iters - cfg.warmup_iters
        )
        expected_lr = cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (
            cfg.learning_rate - cfg.min_lr
        )
        self.assertAlmostEqual(get_lr(mid_decay_it, cfg=cfg), expected_lr)

        # Test end of decay
        self.assertAlmostEqual(get_lr(cfg.lr_decay_iters, cfg=cfg), cfg.min_lr)

    def test_compositional_cyclical_schedule(self):
        """Test the compositional cyclical learning rate schedule."""
        cfg = LRTestConfig(use_cyclical_lr=True)

        # Test warmup phase (should not be affected)
        self.assertAlmostEqual(get_lr(0, cfg=cfg), cfg.learning_rate / cfg.warmup_iters)
        self.assertAlmostEqual(get_lr(49, cfg=cfg), cfg.learning_rate * 0.5)
        self.assertAlmostEqual(get_lr(99, cfg=cfg), cfg.learning_rate)

        # Test modulated decay phase
        it = cfg.warmup_iters + cfg.cyclical_lr_period // 4
        base_lr = get_lr(it, cfg=LRTestConfig(use_cyclical_lr=False))
        expected_lr = base_lr * (1.0 + cfg.cyclical_lr_amplitude)
        self.assertAlmostEqual(get_lr(it, cfg=cfg), expected_lr, delta=1e-5)

        it = cfg.warmup_iters + 3 * cfg.cyclical_lr_period // 4
        base_lr = get_lr(it, cfg=LRTestConfig(use_cyclical_lr=False))
        expected_lr = base_lr * (1.0 - cfg.cyclical_lr_amplitude)
        self.assertAlmostEqual(get_lr(it, cfg=cfg), expected_lr, delta=1e-5)


class TestCyclicalLR(unittest.TestCase):

    def test_cyclical_lr_without_warmup(self):
        """Test cyclical LR with no warmup phase."""
        cfg = CyclicalLRTestConfig(
            warmup_iters=0,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=1000,
            cyclical_lr_period=100,
            cyclical_lr_amplitude=0.2,
        )

        # Test key points in the first cycle
        # At iter 0: should be at peak (sin(0) = 0, so 1 + 0.2*0 = 1.0)
        base_lr_0 = get_lr(
            0,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        cyclical_lr_0 = get_lr(0, cfg=cfg)
        expected_0 = base_lr_0 * (1.0 + cfg.cyclical_lr_amplitude * math.sin(0))
        self.assertAlmostEqual(cyclical_lr_0, expected_0, places=6)

        # At iter 25 (1/4 of cycle): should be at maximum (sin(π/2) = 1)
        base_lr_25 = get_lr(
            25,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        cyclical_lr_25 = get_lr(25, cfg=cfg)
        expected_25 = base_lr_25 * (
            1.0 + cfg.cyclical_lr_amplitude * math.sin(math.pi / 2)
        )
        self.assertAlmostEqual(cyclical_lr_25, expected_25, places=6)

        # At iter 50 (1/2 of cycle): should be back to base (sin(π) = 0)
        base_lr_50 = get_lr(
            50,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        cyclical_lr_50 = get_lr(50, cfg=cfg)
        expected_50 = base_lr_50 * (1.0 + cfg.cyclical_lr_amplitude * math.sin(math.pi))
        self.assertAlmostEqual(cyclical_lr_50, expected_50, places=6)

        # At iter 75 (3/4 of cycle): should be at minimum (sin(3π/2) = -1)
        base_lr_75 = get_lr(
            75,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        cyclical_lr_75 = get_lr(75, cfg=cfg)
        expected_75 = base_lr_75 * (
            1.0 + cfg.cyclical_lr_amplitude * math.sin(3 * math.pi / 2)
        )
        self.assertAlmostEqual(cyclical_lr_75, expected_75, places=6)

        # At iter 100 (full cycle): should be back to base (sin(2π) = 0)
        base_lr_100 = get_lr(
            100,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        cyclical_lr_100 = get_lr(100, cfg=cfg)
        expected_100 = base_lr_100 * (
            1.0 + cfg.cyclical_lr_amplitude * math.sin(2 * math.pi)
        )
        self.assertAlmostEqual(cyclical_lr_100, expected_100, places=6)

    def test_cyclical_lr_with_warmup_composition(self):
        """Test that cyclical LR composes properly with warmup phase."""
        cfg = CyclicalLRTestConfig(
            warmup_iters=100,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=1000,
            cyclical_lr_period=200,
            cyclical_lr_amplitude=0.1,
        )

        # During warmup phase, cyclical LR should NOT be applied
        for warmup_iter in [0, 25, 50, 75, 99]:
            cyclical_lr = get_lr(warmup_iter, cfg=cfg)
            base_lr = get_lr(
                warmup_iter,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=100, lr_decay_iters=1000
                ),
            )
            self.assertAlmostEqual(
                cyclical_lr,
                base_lr,
                places=6,
                msg=f"Warmup iter {warmup_iter}: cyclical LR should equal base LR",
            )

        # After warmup, cyclical LR should be applied
        post_warmup_iter = cfg.warmup_iters + 50  # 50 steps into decay phase
        progress_in_decay = post_warmup_iter - cfg.warmup_iters
        progress_in_cycle = (
            progress_in_decay % cfg.cyclical_lr_period
        ) / cfg.cyclical_lr_period

        base_lr = get_lr(
            post_warmup_iter,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=100, lr_decay_iters=1000
            ),
        )
        cyclical_lr = get_lr(post_warmup_iter, cfg=cfg)

        expected_modulation = 1.0 + cfg.cyclical_lr_amplitude * math.sin(
            2 * math.pi * progress_in_cycle
        )
        expected_lr = base_lr * expected_modulation
        expected_lr = max(cfg.min_lr, expected_lr)  # Clamp to min_lr

        self.assertAlmostEqual(cyclical_lr, expected_lr, places=6)

    def test_cyclical_lr_period_variations(self):
        """Test cyclical LR with different period lengths."""
        base_cfg = CyclicalLRTestConfig(
            warmup_iters=0,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=1000,
            cyclical_lr_amplitude=0.1,
        )

        # Test different periods
        for period in [50, 100, 200, 500]:
            cfg = CyclicalLRTestConfig(**base_cfg.__dict__)
            cfg.cyclical_lr_period = period

            # Test that we complete a full cycle
            iter_quarter = period // 4
            iter_half = period // 2
            iter_three_quarter = 3 * period // 4
            iter_full = period

            # At quarter cycle: should be at maximum
            lr_quarter = get_lr(iter_quarter, cfg=cfg)
            base_lr_quarter = get_lr(
                iter_quarter,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            expected_quarter = base_lr_quarter * (1.0 + cfg.cyclical_lr_amplitude)
            self.assertAlmostEqual(
                lr_quarter,
                expected_quarter,
                places=5,
                msg=f"Period {period}, quarter cycle",
            )

            # At half cycle: should be back to base
            lr_half = get_lr(iter_half, cfg=cfg)
            base_lr_half = get_lr(
                iter_half,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            expected_half = base_lr_half * 1.0  # sin(π) = 0
            self.assertAlmostEqual(
                lr_half, expected_half, places=5, msg=f"Period {period}, half cycle"
            )

            # At three-quarter cycle: should be at minimum
            lr_three_quarter = get_lr(iter_three_quarter, cfg=cfg)
            base_lr_three_quarter = get_lr(
                iter_three_quarter,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            expected_three_quarter = base_lr_three_quarter * (
                1.0 - cfg.cyclical_lr_amplitude
            )
            self.assertAlmostEqual(
                lr_three_quarter,
                expected_three_quarter,
                places=5,
                msg=f"Period {period}, three-quarter cycle",
            )

    def test_cyclical_lr_amplitude_variations(self):
        """Test cyclical LR with different amplitude values."""
        base_cfg = CyclicalLRTestConfig(
            warmup_iters=0,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=1000,
            cyclical_lr_period=100,
        )

        # Test different amplitudes
        for amplitude in [0.0, 0.05, 0.1, 0.2, 0.5]:
            cfg = CyclicalLRTestConfig(**base_cfg.__dict__)
            cfg.cyclical_lr_amplitude = amplitude

            # Test at peak (quarter cycle)
            peak_iter = cfg.cyclical_lr_period // 4
            lr_peak = get_lr(peak_iter, cfg=cfg)
            base_lr_peak = get_lr(
                peak_iter,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            expected_peak = base_lr_peak * (1.0 + amplitude)
            self.assertAlmostEqual(
                lr_peak, expected_peak, places=6, msg=f"Amplitude {amplitude}, peak"
            )

            # Test at trough (three-quarter cycle)
            trough_iter = 3 * cfg.cyclical_lr_period // 4
            lr_trough = get_lr(trough_iter, cfg=cfg)
            base_lr_trough = get_lr(
                trough_iter,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            expected_trough = base_lr_trough * (1.0 - amplitude)
            expected_trough = max(cfg.min_lr, expected_trough)  # Should be clamped
            self.assertAlmostEqual(
                lr_trough,
                expected_trough,
                places=6,
                msg=f"Amplitude {amplitude}, trough",
            )

    def test_cyclical_lr_min_lr_clamping(self):
        """Test that cyclical LR is properly clamped to min_lr."""
        cfg = CyclicalLRTestConfig(
            warmup_iters=0,
            learning_rate=1e-3,
            min_lr=5e-4,  # Higher min_lr
            lr_decay_iters=1000,
            cyclical_lr_period=100,
            cyclical_lr_amplitude=0.8,  # Large amplitude to force clamping
        )

        # Test at trough where clamping should occur
        trough_iter = 3 * cfg.cyclical_lr_period // 4
        lr_trough = get_lr(trough_iter, cfg=cfg)

        # Should be clamped to min_lr
        self.assertGreaterEqual(lr_trough, cfg.min_lr)

        # If unclamped value would be below min_lr, should equal min_lr
        base_lr_trough = get_lr(
            trough_iter,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        unclamped_trough = base_lr_trough * (1.0 - cfg.cyclical_lr_amplitude)

        if unclamped_trough < cfg.min_lr:
            self.assertAlmostEqual(lr_trough, cfg.min_lr, places=6)
        else:
            self.assertAlmostEqual(lr_trough, unclamped_trough, places=6)

    def test_cyclical_lr_post_decay_phase(self):
        """Test cyclical LR behavior after decay phase ends."""
        cfg = CyclicalLRTestConfig(
            warmup_iters=100,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=500,  # Short decay phase
            cyclical_lr_period=100,
            cyclical_lr_amplitude=0.1,
        )

        # After decay phase, base LR should be min_lr
        post_decay_iter = cfg.lr_decay_iters + 100
        lr_post_decay = get_lr(post_decay_iter, cfg=cfg)

        # Should be exactly min_lr (no modulation should occur)
        self.assertAlmostEqual(lr_post_decay, cfg.min_lr, places=6)

    def test_cyclical_lr_mathematical_correctness(self):
        """Test mathematical correctness of the cyclical pattern."""
        cfg = CyclicalLRTestConfig(
            warmup_iters=0,
            learning_rate=1e-3,
            min_lr=1e-4,
            lr_decay_iters=1000,
            cyclical_lr_period=100,
            cyclical_lr_amplitude=0.1,
        )

        # Test multiple cycles to ensure pattern repeats correctly
        for cycle in range(3):
            cycle_start = cycle * cfg.cyclical_lr_period

            # Test that pattern repeats every cycle
            for offset in [0, 25, 50, 75]:
                iter1 = cycle_start + offset
                iter2 = cycle_start + cfg.cyclical_lr_period + offset

                lr1 = get_lr(iter1, cfg=cfg)
                lr2 = get_lr(iter2, cfg=cfg)

                # Get base LRs (they will be different due to cosine decay)
                base_lr1 = get_lr(
                    iter1,
                    cfg=CyclicalLRTestConfig(
                        use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                    ),
                )
                base_lr2 = get_lr(
                    iter2,
                    cfg=CyclicalLRTestConfig(
                        use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                    ),
                )

                # The modulation factor should be the same
                modulation1 = lr1 / base_lr1 if base_lr1 > 0 else 1.0
                modulation2 = lr2 / base_lr2 if base_lr2 > 0 else 1.0

                self.assertAlmostEqual(
                    modulation1,
                    modulation2,
                    places=5,
                    msg=f"Cycle {cycle}, offset {offset}: modulation should repeat",
                )

    def test_cyclical_lr_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small amplitude
        cfg_small = CyclicalLRTestConfig(warmup_iters=0, cyclical_lr_amplitude=0.001)

        lr_base = get_lr(
            0,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        lr_cyclical = get_lr(0, cfg=cfg_small)

        # Should be very close to base LR
        self.assertAlmostEqual(lr_cyclical, lr_base, places=3)

        # Test with amplitude = 0 (should be same as base)
        cfg_zero = CyclicalLRTestConfig(warmup_iters=0, cyclical_lr_amplitude=0.0)

        for iter_num in [0, 50, 100, 200]:
            lr_base = get_lr(
                iter_num,
                cfg=CyclicalLRTestConfig(
                    use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
                ),
            )
            lr_cyclical = get_lr(iter_num, cfg=cfg_zero)
            self.assertAlmostEqual(lr_cyclical, lr_base, places=6)

        # Test with very short period
        cfg_short = CyclicalLRTestConfig(warmup_iters=0, cyclical_lr_period=4)

        # Should still follow sinusoidal pattern
        lr_0 = get_lr(0, cfg=cfg_short)
        lr_1 = get_lr(1, cfg=cfg_short)  # Quarter cycle
        lr_2 = get_lr(2, cfg=cfg_short)  # Half cycle
        lr_3 = get_lr(3, cfg=cfg_short)  # Three-quarter cycle
        lr_4 = get_lr(4, cfg=cfg_short)  # Full cycle

        base_0 = get_lr(
            0,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        base_1 = get_lr(
            1,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        base_2 = get_lr(
            2,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        base_3 = get_lr(
            3,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )
        base_4 = get_lr(
            4,
            cfg=CyclicalLRTestConfig(
                use_cyclical_lr=False, warmup_iters=0, lr_decay_iters=1000
            ),
        )

        # Check pattern
        self.assertAlmostEqual(lr_0, base_0 * 1.0, places=5)  # sin(0) = 0
        self.assertAlmostEqual(
            lr_1, base_1 * (1.0 + cfg_short.cyclical_lr_amplitude), places=5
        )  # sin(π/2) = 1
        self.assertAlmostEqual(lr_2, base_2 * 1.0, places=5)  # sin(π) = 0
        self.assertAlmostEqual(
            lr_3, base_3 * (1.0 - cfg_short.cyclical_lr_amplitude), places=5
        )  # sin(3π/2) = -1
        self.assertAlmostEqual(lr_4, base_4 * 1.0, places=5)  # sin(2π) = 0


class TestLearningRateLogging(unittest.TestCase):
    """Ensure that the learning-rate schedule and logging behave as expected."""

    def setUp(self):
        self.cfg = _LRLoggingTestConfig()
        # Simple model with a single parameter so the optimizer is lightweight.
        self.model = nn.Linear(4, 2)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate)

    def test_optimizer_lr_matches_schedule(self):
        """Verify that the LR we place into the optimizer matches get_lr()."""
        for iter_num in range(self.cfg.max_iters + 1):
            scheduled_lr = get_lr(iter_num, cfg=self.cfg)
            # Simulate the update performed in the training loop.
            for pg in self.optimizer.param_groups:
                pg["lr"] = scheduled_lr
            # The logged value should be pulled from the optimizer after update.
            logged_lr = self.optimizer.param_groups[0]["lr"]
            self.assertAlmostEqual(logged_lr, scheduled_lr, places=8)

    def test_schedule_is_not_strictly_linear(self):
        """Confirm the LR schedule contains both increasing and decreasing phases."""
        lrs = [get_lr(i, cfg=self.cfg) for i in range(self.cfg.max_iters + 1)]
        # Ensure warmup increases.
        self.assertGreater(lrs[self.cfg.warmup_iters], lrs[0])
        # Ensure decay phase decreases after warmup.
        self.assertLess(lrs[-1], lrs[self.cfg.warmup_iters])


if __name__ == "__main__":
    unittest.main()
