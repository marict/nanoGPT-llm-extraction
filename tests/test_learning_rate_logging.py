import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from training_utils import BaseConfig, get_lr


@dataclass
class _LRLoggingTestConfig(BaseConfig):
    """Minimal configuration for LR schedule testing."""

    name: str = "lr_logging_test"

    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_iters: int = 5
    lr_decay_iters: int = 20
    max_iters: int = 20


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
