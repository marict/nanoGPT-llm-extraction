import unittest
from unittest import mock

from dag_model import GPT, GPTConfig
from evaluation import evaluate_math


class TestMathEvalSkip(unittest.TestCase):
    """Ensure evaluate_math gracefully skips when max_examples is 0."""

    def test_skip_evaluation_when_zero_examples(self):
        # Create a minimal GPT model
        config = GPTConfig(
            n_layer=1,
            n_head=1,
            n_embd=8,
            vocab_size=100,
            block_size=8,
            bias=False,
            dag_depth=0,
        )
        model = GPT(config)

        # Patch run_math_eval.run_eval to assert it is NOT called
        with mock.patch("run_math_eval.run_eval") as mock_run_eval:
            scores = evaluate_math(model, "cpu", max_examples=0)
            mock_run_eval.assert_not_called()
            self.assertEqual(scores, {}, "Expected empty dict when skipping math eval")


if __name__ == "__main__":
    unittest.main()
