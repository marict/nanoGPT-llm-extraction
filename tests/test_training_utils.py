import math
import unittest
from dataclasses import dataclass

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
