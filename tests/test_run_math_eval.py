"""Unit tests for run_math_eval.py"""

import sys
from pathlib import Path
from unittest import TestCase, mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_common import MockModel, mock_decode, mock_encode

import run_math_eval


class DummyModel(torch.nn.Module):
    """Fake model with encode/decode/generate interface."""

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([1, 2, 3])

    def decode(self, ids) -> str:
        return "42"

    def generate(self, ids, max_new_tokens: int, temperature: float = 1.0, top_k=None):
        return ids.new_tensor([[1, 2, 3, 4]])


class DummyModelWithConfig:
    """Minimal model for testing tokenizer attachment."""

    def __init__(self):
        self.config = type("Config", (), {"vocab_size": 50257})()


class TestRunMathEval(TestCase):
    """Streamlined tests for math evaluation functionality."""

    def test_extract_number_comprehensive(self):
        """Test number extraction for various formats."""
        test_cases = [
            ("The answer is #### 42", "42"),
            ("#### 100", "100"),
            ("Answer: #### -5", "-5"),
            ("No answer here", ""),
            ("Multiple #### 10 and #### 20", "10"),  # First match
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = run_math_eval._extract_number(text)
                self.assertEqual(result, expected)

    def test_tokenizer_attachment_integration(self):
        """Test tokenizer attachment with different scenarios."""
        model = DummyModelWithConfig()

        # Test with GPT-2 meta file
        with mock.patch("builtins.open", mock.mock_open()):
            with mock.patch("pickle.load") as mock_pickle_load:
                with mock.patch("pathlib.Path.exists", return_value=True):
                    with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
                        mock_pickle_load.return_value = {
                            "tokenizer": "gpt2",
                            "vocab_size": 50257,
                        }

                        mock_encoder = mock.MagicMock()
                        mock_encoder.encode.return_value = [1, 2, 3]
                        mock_encoder.decode.return_value = "test"
                        mock_get_encoding.return_value = mock_encoder

                        run_math_eval._attach_tokenizer_methods(model, "cpu")

        # Verify methods were attached
        self.assertTrue(hasattr(model, "encode"))
        self.assertTrue(hasattr(model, "decode"))

        # Test without meta file (defaults to GPT-2)
        model2 = DummyModelWithConfig()
        with mock.patch("pathlib.Path.exists", return_value=False):
            with mock.patch("pathlib.Path.glob", return_value=[]):
                with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
                    mock_encoder = mock.MagicMock()
                    mock_encoder.encode.return_value = [1, 2, 3]
                    mock_encoder.decode.return_value = "test"
                    mock_get_encoding.return_value = mock_encoder

                    run_math_eval._attach_tokenizer_methods(model2, "cpu")

        self.assertTrue(hasattr(model2, "encode"))
        self.assertTrue(hasattr(model2, "decode"))

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_functionality(self, mock_load_ds):
        """Test the main evaluation functionality."""
        # Create mock dataset
        fake_ds = [
            {"question": "What is 1+41?", "answer": "#### 42"},
            {"question": "What is 10+10?", "answer": "#### 20"},
        ]

        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__ = lambda self: iter(fake_ds)
        mock_dataset.__len__ = lambda self: len(fake_ds)
        mock_dataset.select.return_value = mock_dataset
        mock_load_ds.return_value = mock_dataset

        # Test with mock model
        model = MockModel()
        model.encode = mock_encode
        model.decode = mock_decode

        # Run evaluation
        scores = run_math_eval.run_eval(
            model, device="cpu", tasks=["gsm8k"], max_examples=2
        )

        # Verify results
        self.assertIn("gsm8k", scores)
        self.assertTrue(0 <= scores["gsm8k"] <= 1)

    def test_dataset_loading_error_handling(self):
        """Test error handling in dataset loading."""
        # Test dataset loading with error
        with mock.patch(
            "datasets.load_dataset", side_effect=Exception("Network error")
        ):
            with self.assertRaises(Exception):
                run_math_eval._load_dataset("gsm8k")


if __name__ == "__main__":
    import unittest

    unittest.main()
