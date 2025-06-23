"""Unit tests for run_math_eval.py"""

from types import SimpleNamespace
from unittest import TestCase, mock

import torch

import run_math_eval


class DummyModel(torch.nn.Module):
    """Fake model with encode/decode/generate interface."""

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([1, 2, 3])

    def decode(self, ids) -> str:
        return "42"

    def generate(self, ids, max_new_tokens: int, temperature: float = 1.0, top_k=None):
        return ids.new_tensor([[1, 2, 3, 4]])


class DummyModelWithConfig(torch.nn.Module):
    """Fake model with config attribute."""

    def __init__(self, dataset: str = "shakespeare"):
        super().__init__()
        self.config = SimpleNamespace(dataset=dataset)

    def generate(self, ids, max_new_tokens: int, temperature: float = 1.0, top_k=None):
        return ids.new_tensor([[1, 2, 3, 4]])

    def eval(self):
        return self

    def train(self):
        return self

    def to(self, device):
        return self


class TestRunMathEval(TestCase):
    def test_extract_number(self):
        self.assertEqual(run_math_eval._extract_number("Answer: 42"), "42")
        self.assertEqual(run_math_eval._extract_number("no numbers"), "")
        self.assertEqual(run_math_eval._extract_number("The answer is -5.5"), "-5.5")
        self.assertEqual(run_math_eval._extract_number("Numbers: 123, 456"), "123")
        self.assertEqual(run_math_eval._extract_number(""), "")

    def test_make_prompt(self):
        # Test GSM8K prompt formatting
        gsm8k_ex = {"question": "What is 2+2?"}
        prompt = run_math_eval._make_prompt("gsm8k", gsm8k_ex)
        self.assertEqual(prompt, "What is 2+2?\nAnswer:")

        # Test SVAMP prompt formatting
        svamp_ex = {"Body": "There are 5 apples.", "Question": "How many fruits?"}
        prompt = run_math_eval._make_prompt("svamp", svamp_ex)
        self.assertEqual(prompt, "There are 5 apples. How many fruits?\nAnswer:")

        # Test unsupported task
        with self.assertRaises(ValueError):
            run_math_eval._make_prompt("unsupported_task", {})

    def test_gold_answer(self):
        # Test GSM8K answer extraction
        gsm8k_ex = {"answer": "Let me solve this step by step.\n#### 42"}
        answer = run_math_eval._gold_answer("gsm8k", gsm8k_ex)
        self.assertEqual(answer, "42")

        # Test SVAMP answer extraction
        svamp_ex = {"Answer": 42}
        answer = run_math_eval._gold_answer("svamp", svamp_ex)
        self.assertEqual(answer, "42")

        svamp_ex_str = {"Answer": "42"}
        answer = run_math_eval._gold_answer("svamp", svamp_ex_str)
        self.assertEqual(answer, "42")

        # Test unsupported task
        with self.assertRaises(ValueError):
            run_math_eval._gold_answer("unsupported_task", {})

    @mock.patch("run_math_eval.datasets.load_dataset")
    def test_load_dataset(self, mock_load):
        mock_dataset = [{"question": "test", "answer": "#### 1"}]
        mock_load.return_value = mock_dataset

        # Test GSM8K dataset loading
        result = run_math_eval._load_dataset("gsm8k")
        mock_load.assert_called_with("gsm8k", "main", split="test")
        self.assertEqual(result, mock_dataset)

        # Test SVAMP dataset loading
        result = run_math_eval._load_dataset("svamp")
        mock_load.assert_called_with("svamp", split="test")
        self.assertEqual(result, mock_dataset)

        # Test unsupported task
        with self.assertRaises(ValueError):
            run_math_eval._load_dataset("unsupported_task")

    def test_attach_tokenizer_methods_char_level(self):
        """Test tokenizer attachment with character-level tokenizer."""
        model = DummyModelWithConfig()

        # Mock the function to directly test the tokenizer attachment
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            with mock.patch("pickle.load") as mock_pickle_load:
                with mock.patch("pathlib.Path.exists") as mock_exists:
                    mock_exists.return_value = True
                    mock_pickle_load.return_value = {"stoi": {"a": 0}, "itos": {0: "a"}}

                    run_math_eval._attach_tokenizer_methods(model, "cpu")

        # Check that methods were attached
        self.assertTrue(hasattr(model, "encode"))
        self.assertTrue(hasattr(model, "decode"))

    def test_attach_tokenizer_methods_gpt2_meta(self):
        """Test tokenizer attachment with GPT-2 meta file."""
        model = DummyModelWithConfig()

        # Mock the function to directly test the tokenizer attachment
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            with mock.patch("pickle.load") as mock_pickle_load:
                with mock.patch("pathlib.Path.exists") as mock_exists:
                    with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
                        mock_exists.return_value = True
                        mock_pickle_load.return_value = {
                            "tokenizer": "gpt2",
                            "vocab_size": 50257,
                        }

                        # Mock the encoder object
                        mock_encoder = mock.MagicMock()
                        mock_encoder.encode.return_value = [1, 2, 3]
                        mock_encoder.decode.return_value = "test"
                        mock_get_encoding.return_value = mock_encoder

                        run_math_eval._attach_tokenizer_methods(model, "cpu")

        # Check that methods were attached
        self.assertTrue(hasattr(model, "encode"))
        self.assertTrue(hasattr(model, "decode"))

    def test_attach_tokenizer_methods_no_meta(self):
        """Test tokenizer attachment with no meta file (defaults to GPT-2)."""
        model = DummyModelWithConfig()

        # Mock to simulate no meta files found
        with mock.patch("pathlib.Path.exists") as mock_exists:
            with mock.patch("pathlib.Path.glob") as mock_glob:
                with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
                    mock_exists.return_value = False
                    mock_glob.return_value = []

                    # Mock the encoder object
                    mock_encoder = mock.MagicMock()
                    mock_encoder.encode.return_value = [1, 2, 3]
                    mock_encoder.decode.return_value = "test"
                    mock_get_encoding.return_value = mock_encoder

                    run_math_eval._attach_tokenizer_methods(model, "cpu")

        # Check that methods were attached (should default to GPT-2)
        self.assertTrue(hasattr(model, "encode"))
        self.assertTrue(hasattr(model, "decode"))

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_gsm8k(self, mock_load_ds):
        fake_ds = [
            {"question": "What is 1+41?", "answer": "#### 42"},
            {"question": "What is 10+10?", "answer": "#### 42"},
        ]

        # Mock dataset to have select method
        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__ = lambda self: iter(fake_ds)
        mock_dataset.__len__ = lambda self: len(fake_ds)
        mock_dataset.select.return_value = mock_dataset
        mock_load_ds.return_value = mock_dataset

        model = DummyModel()
        scores = run_math_eval.run_eval(model, device="cpu", tasks=["gsm8k"])
        self.assertIn("gsm8k", scores)
        self.assertTrue(0 <= scores["gsm8k"] <= 1)

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_svamp(self, mock_load_ds):
        fake_ds = [
            {
                "Body": "There are 40 apples.",
                "Question": "If you add 2, how many?",
                "Answer": 42,
            },
            {
                "Body": "You have 20 pens.",
                "Question": "If you get 22 more?",
                "Answer": 42,
            },
        ]

        # Mock dataset to have select method
        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__ = lambda self: iter(fake_ds)
        mock_dataset.__len__ = lambda self: len(fake_ds)
        mock_dataset.select.return_value = mock_dataset
        mock_load_ds.return_value = mock_dataset

        model = DummyModel()
        scores = run_math_eval.run_eval(model, device="cpu", tasks=["svamp"])
        self.assertIn("svamp", scores)
        self.assertTrue(0 <= scores["svamp"] <= 1)

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_max_examples(self, mock_load_ds):
        fake_ds = [
            {"question": "What is 1+41?", "answer": "#### 42"},
            {"question": "What is 10+10?", "answer": "#### 42"},
            {"question": "What is 5+5?", "answer": "#### 10"},
        ]

        # Mock dataset to have select method
        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__ = lambda self: iter(fake_ds[:2])  # Only return first 2
        mock_dataset.__len__ = lambda self: 2
        mock_dataset.select.return_value = mock_dataset
        mock_load_ds.return_value = (
            mock_dataset  # Return the mock dataset, not the list
        )

        model = DummyModel()
        scores = run_math_eval.run_eval(
            model, device="cpu", tasks=["gsm8k"], max_examples=2
        )

        # Verify select was called with the right range
        mock_dataset.select.assert_called_once_with(range(2))

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_dataset_error(self, mock_load_ds):
        """Test error handling when dataset loading fails."""
        mock_load_ds.side_effect = Exception("Dataset not found")

        model = DummyModel()
        scores = run_math_eval.run_eval(model, device="cpu", tasks=["gsm8k"])

        # Should return 0.0 for failed tasks
        self.assertEqual(scores["gsm8k"], 0.0)

    @mock.patch("run_math_eval._load_dataset")
    def test_run_eval_processing_error(self, mock_load_ds):
        """Test error handling when individual examples fail."""
        fake_ds = [
            {"question": "What is 1+41?", "answer": "#### 42"},
        ]

        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__ = lambda self: iter(fake_ds)
        mock_dataset.__len__ = lambda self: len(fake_ds)
        mock_dataset.select.return_value = mock_dataset
        mock_load_ds.return_value = mock_dataset

        # Create model that will fail on generate
        class FailingModel(DummyModel):
            def generate(self, *args, **kwargs):
                raise Exception("Generation failed")

        model = FailingModel()
        scores = run_math_eval.run_eval(model, device="cpu", tasks=["gsm8k"])

        # Should handle error gracefully
        self.assertEqual(scores["gsm8k"], 0.0)

    def test_run_eval_default_tasks(self):
        """Test that default tasks are used when none specified."""
        with mock.patch("run_math_eval._load_dataset") as mock_load_ds:
            mock_dataset = mock.MagicMock()
            mock_dataset.__iter__ = lambda self: iter([])
            mock_dataset.__len__ = lambda self: 0
            mock_dataset.select.return_value = mock_dataset
            mock_load_ds.return_value = mock_dataset

            model = DummyModel()
            scores = run_math_eval.run_eval(model, device="cpu")

            # Should have evaluated both default tasks
            self.assertIn("gsm8k", scores)
            self.assertIn("svamp", scores)

    def test_run_eval_model_state_management(self):
        """Test that model is set to eval mode during evaluation and back to train."""

        class StatefulModel(DummyModel):
            def __init__(self):
                super().__init__()
                self.training = True
                self._device = None

            def eval(self):
                self.training = False
                return self

            def train(self):
                self.training = True
                return self

            def to(self, device):
                self._device = device
                return self

        with mock.patch("run_math_eval._load_dataset") as mock_load_ds:
            mock_dataset = mock.MagicMock()
            mock_dataset.__iter__ = lambda self: iter([])
            mock_dataset.__len__ = lambda self: 0
            mock_dataset.select.return_value = mock_dataset
            mock_load_ds.return_value = mock_dataset

            model = StatefulModel()
            self.assertTrue(model.training)  # Initially in training mode

            run_math_eval.run_eval(model, device="cpu", tasks=["gsm8k"])

            # Should be back in training mode
            self.assertTrue(model.training)
            # Should have been moved to correct device
            self.assertEqual(model._device, "cpu")

    @mock.patch("sys.argv", ["run_math_eval.py", "--help"])
    def test_main_help(self):
        """Test that the main function has proper CLI help."""
        with self.assertRaises(SystemExit):
            run_math_eval.main()

    @mock.patch("torch.load")
    @mock.patch(
        "sys.argv", ["run_math_eval.py", "fake_checkpoint.pt", "--max-examples", "1"]
    )
    def test_main_function_gpt(self, mock_torch_load):
        """Test main function with GPT model."""
        # Mock checkpoint loading
        mock_checkpoint = {
            "model_args": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "vocab_size": 50257,
            },
            "model": {},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock model creation and evaluation
        with mock.patch("run_math_eval.run_eval") as mock_run_eval:
            with mock.patch("model.GPT") as mock_gpt_class:
                mock_model = DummyModel()
                mock_gpt_class.return_value = mock_model
                mock_run_eval.return_value = {"gsm8k": 0.5, "svamp": 0.3}

                run_math_eval.main()

                # Verify model creation and evaluation were called
                mock_gpt_class.assert_called_once()
                mock_run_eval.assert_called_once()

    @mock.patch("torch.load")
    @mock.patch(
        "sys.argv", ["run_math_eval.py", "fake_dag_checkpoint.pt", "--tasks", "gsm8k"]
    )
    def test_main_function_daggpt(self, mock_torch_load):
        """Test main function with DAGGPT model."""
        # Mock checkpoint loading
        mock_checkpoint = {
            "model_args": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "vocab_size": 50257,
                "dag_depth": 3,
            },
            "model": {},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock model creation and evaluation
        with mock.patch("run_math_eval.run_eval") as mock_run_eval:
            with mock.patch("dag_model.DAGGPT") as mock_daggpt_class:
                mock_model = DummyModel()
                mock_daggpt_class.return_value = mock_model
                mock_run_eval.return_value = {"gsm8k": 0.4}

                run_math_eval.main()

                # Verify DAGGPT model creation and evaluation were called
                mock_daggpt_class.assert_called_once()
                mock_run_eval.assert_called_once()
