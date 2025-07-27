"""Comprehensive tests for tokenization and final_token_pos calculation.

These tests ensure that:
1. tokenize_texts() works correctly with padding
2. final_token_pos is calculated correctly from real mathematical expressions
3. The end-to-end pipeline from expression → tokenization → loss masking works
4. Edge cases and boundary conditions are handled properly
"""

import pytest
import tiktoken
import torch

from data.dagset.streaming import (
    create_dag_structure_dataloaders,
    generate_single_dag_example,
)
from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss, tokenize_texts


class TestTokenizeTexts:
    """Test the tokenize_texts function directly."""

    def test_tokenize_texts_basic_functionality(self):
        """Test basic tokenization with padding."""
        texts = ["2 + 3", "x * y", "10 / 5"]
        block_size = 10
        device = "cpu"

        result = tokenize_texts(texts, block_size, device)

        # Check output shape
        assert result.shape == (3, block_size)
        assert result.device.type == device

        # Check that each sequence is padded to block_size
        encoding = tiktoken.get_encoding("gpt2")
        for i, text in enumerate(texts):
            tokens = encoding.encode_ordinary(text)
            # Check that original tokens are preserved
            assert result[i, : len(tokens)].tolist() == tokens
            # Check that padding is zeros
            assert (result[i, len(tokens) :] == 0).all()

    def test_tokenize_texts_empty_string(self):
        """Test tokenization with empty strings."""
        texts = ["", "2 + 3", ""]
        block_size = 5
        device = "cpu"

        result = tokenize_texts(texts, block_size, device)

        # Empty strings should be all padding
        assert (result[0] == 0).all()
        assert (result[2] == 0).all()

        # Non-empty string should have content + padding
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode_ordinary("2 + 3")
        assert result[1, : len(tokens)].tolist() == tokens
        assert (result[1, len(tokens) :] == 0).all()

    def test_tokenize_texts_sequence_too_long(self):
        """Test behavior when text is longer than block_size."""
        # Create a long expression
        long_text = " + ".join(str(i) for i in range(20))  # "0 + 1 + 2 + ... + 19"
        texts = [long_text]
        block_size = 5  # Much smaller than needed
        device = "cpu"

        result = tokenize_texts(texts, block_size, device)

        # Should truncate to block_size
        assert result.shape == (1, block_size)

        # All positions should be filled (no padding)
        assert (result[0] != 0).all()

    def test_tokenize_texts_exact_length(self):
        """Test when text exactly fills block_size."""
        # Find a text that tokenizes to exactly 3 tokens
        encoding = tiktoken.get_encoding("gpt2")

        # "x+y" typically tokenizes to exactly 3 tokens
        text = "x+y"
        tokens = encoding.encode_ordinary(text)
        block_size = len(tokens)

        result = tokenize_texts([text], block_size, "cpu")

        # Should exactly fill block_size with no padding
        assert result.shape == (1, block_size)
        assert result[0].tolist() == tokens


class TestFinalTokenPosCalculation:
    """Test final_token_pos calculation from real mathematical expressions."""

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_final_token_pos_from_dag_generation(self, depth):
        """Test that generated DAG examples have correct final_token_pos."""
        example = generate_single_dag_example(
            depth=depth, seed=42, max_digits=2, max_decimal_places=1
        )

        # Manually calculate expected final_token_pos
        encoding = tiktoken.get_encoding("gpt2")
        expected_tokens = encoding.encode_ordinary(example.text)
        expected_final_pos = len(expected_tokens) - 1 if expected_tokens else 0

        # Check that the example has correct final_token_pos
        assert example.structure_dict["final_token_pos"] == expected_final_pos

        # Verify the text is not empty for valid depths
        assert example.text.strip() != ""
        assert expected_final_pos >= 0

    def test_final_token_pos_various_expressions(self):
        """Test final_token_pos calculation for various mathematical expressions."""
        test_cases = [
            ("2", 0),  # Single token
            ("2 + 3", None),  # Let the tokenizer determine
            ("x * y / z", None),
            ("(10 + 5) * 2", None),
            ("3.14159", None),
        ]

        encoding = tiktoken.get_encoding("gpt2")

        for text, expected_pos in test_cases:
            tokens = encoding.encode_ordinary(text)
            actual_final_pos = len(tokens) - 1 if tokens else 0

            if expected_pos is not None:
                assert (
                    actual_final_pos == expected_pos
                ), f"Text '{text}' tokens: {tokens}"
            else:
                # Just verify it's reasonable
                assert actual_final_pos >= 0
                assert (
                    actual_final_pos < len(tokens) if tokens else actual_final_pos == 0
                )

    def test_final_token_pos_edge_cases(self):
        """Test edge cases for final_token_pos calculation."""
        encoding = tiktoken.get_encoding("gpt2")

        # Empty string
        tokens = encoding.encode_ordinary("")
        final_pos = len(tokens) - 1 if tokens else 0
        assert final_pos == 0  # Empty should give position 0

        # Single character
        tokens = encoding.encode_ordinary("x")
        final_pos = len(tokens) - 1 if tokens else 0
        assert final_pos >= 0

        # Unicode/special characters
        tokens = encoding.encode_ordinary("π²")
        final_pos = len(tokens) - 1 if tokens else 0
        assert final_pos >= 0


class TestEndToEndTokenizationPipeline:
    """Test the complete pipeline from mathematical expression to loss calculation."""

    def test_end_to_end_tokenization_and_loss(self):
        """Test complete pipeline: expression → tokenization → final_token_pos → loss."""
        # Generate a real DAG example
        example = generate_single_dag_example(
            depth=2, seed=123, max_digits=2, max_decimal_places=1, train=True
        )

        # Verify the example has all required components
        assert hasattr(example, "text")
        assert hasattr(example, "structure_dict")
        assert "final_token_pos" in example.structure_dict

        # Tokenize the text
        block_size = 20
        device = "cpu"
        tokenized = tokenize_texts([example.text], block_size, device)

        # Verify tokenization is consistent with final_token_pos
        encoding = tiktoken.get_encoding("gpt2")
        expected_tokens = encoding.encode_ordinary(example.text)

        if expected_tokens:
            # Check that tokenized version matches expected
            assert tokenized[0, : len(expected_tokens)].tolist() == expected_tokens
            # Check that final_token_pos is correct
            assert example.structure_dict["final_token_pos"] == len(expected_tokens) - 1

    def test_batch_tokenization_consistency(self):
        """Test that batch tokenization gives consistent final_token_pos values."""
        # Generate multiple examples
        examples = [
            generate_single_dag_example(depth=2, seed=i, train=True)
            for i in range(42, 47)  # 5 examples
        ]

        texts = [ex.text for ex in examples]
        block_size = 50
        device = "cpu"

        # Batch tokenize
        tokenized_batch = tokenize_texts(texts, block_size, device)

        # Verify each example individually
        encoding = tiktoken.get_encoding("gpt2")
        for i, (text, example) in enumerate(zip(texts, examples)):
            expected_tokens = encoding.encode_ordinary(text)
            expected_final_pos = len(expected_tokens) - 1 if expected_tokens else 0

            # Check tokenization
            if expected_tokens:
                assert (
                    tokenized_batch[i, : len(expected_tokens)].tolist()
                    == expected_tokens
                )

            # Check final_token_pos from structure_dict
            assert example.structure_dict["final_token_pos"] == expected_final_pos


class TestLossMaskingIntegration:
    """Test that loss masking works correctly with real tokenized data."""

    def test_loss_masking_with_real_expressions(self):
        """Test loss calculation with real mathematical expressions and their final_token_pos."""
        # Create a small batch of real examples
        batch_size = 3
        examples = [
            generate_single_dag_example(depth=2, seed=100 + i, train=True)
            for i in range(batch_size)
        ]

        # Extract texts and final_token_pos
        texts = [ex.text for ex in examples]
        final_token_positions = [
            ex.structure_dict["final_token_pos"] for ex in examples
        ]

        # Tokenize
        block_size = 30
        device = "cpu"
        input_tokens = tokenize_texts(texts, block_size, device)

        # Create dummy model predictions (realistic shapes)
        B, T = input_tokens.shape
        N = 3  # nodes
        D = 4  # digits
        depth = 2
        base = 10
        n_ops = 5  # Need 5 ops: add, subtract, multiply, divide, identity

        # Create realistic prediction tensors
        pred_sign_logits = torch.randn(B, T, N)
        pred_digit_logits = torch.randn(B, T, N, D, base)
        pred_op_logits = torch.randn(B, T, depth, n_ops)
        pred_statistics = {
            "initial": torch.randn(B, T, 6),
            "intermediate": torch.randn(B, T, 6),
            "final": torch.randn(B, T, 6),
        }

        # Create dummy targets (batch format without sequence dimension)
        target_sgn = torch.randn(B, N)
        target_digits = torch.zeros(B, N, D, base)
        target_digits[:, :, :, 5] = 1.0  # All digits are "5"
        target_ops = torch.zeros(B, depth, n_ops)
        target_ops[:, :, 0] = 1.0  # All operations are first op
        target_initial_values = torch.randn(B, N)
        target_final_exec = torch.randn(B)
        target_statistics = {
            "initial": torch.randn(B, 6),
            "intermediate": torch.randn(B, 6),
            "final": torch.randn(B, 6),
        }

        final_token_pos = torch.tensor(final_token_positions, dtype=torch.long)

        # Create config
        cfg = DAGTrainConfig()
        cfg.max_digits = D - 1  # One less than digit tensor size
        cfg.max_decimal_places = 1
        cfg.base = base

        # Compute loss - this should work without errors
        losses = compute_dag_structure_loss(
            pred_sign_logits,
            pred_digit_logits,
            pred_op_logits,
            pred_statistics,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            target_statistics,
            final_token_pos,
            cfg,
            uncertainty_params=torch.zeros(6),
        )

        # Verify loss computation succeeded
        assert "total_loss" in losses
        assert torch.isfinite(losses["total_loss"])
        assert losses["total_loss"].item() >= 0

    def test_dataloader_integration(self):
        """Test that the data loader produces correctly tokenized data with valid final_token_pos."""
        train_loader, _ = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=4,
            max_depth=2,
            seed=42,
            max_digits=2,
            max_decimal_places=1,
            base=10,
            allowed_operations=["add", "subtract"],
        )

        # Get one batch
        texts, structures, meta = next(train_loader)

        # Verify batch structure
        assert len(texts) == 4
        assert "final_token_pos" in structures
        assert structures["final_token_pos"].shape == (4,)

        # Verify each text has correct final_token_pos
        encoding = tiktoken.get_encoding("gpt2")
        for i, text in enumerate(texts):
            tokens = encoding.encode_ordinary(text)
            expected_final_pos = len(tokens) - 1 if tokens else 0
            actual_final_pos = structures["final_token_pos"][i].item()

            assert actual_final_pos == expected_final_pos, (
                f"Text {i}: '{text}' -> tokens: {tokens} -> "
                f"expected final_pos: {expected_final_pos}, got: {actual_final_pos}"
            )


class TestBoundaryConditions:
    """Test boundary conditions and error cases."""

    def test_final_token_pos_bounds_checking(self):
        """Test that final_token_pos bounds checking works in loss function."""
        B, T, N, D, depth = 2, 5, 2, 3, 1
        base, n_ops = 10, 5  # Need 5 ops: add, subtract, multiply, divide, identity

        # Create dummy tensors
        pred_sign_logits = torch.randn(B, T, N)
        pred_digit_logits = torch.randn(B, T, N, D, base)
        pred_op_logits = torch.randn(B, T, depth, n_ops)
        pred_statistics = {
            "initial": torch.randn(B, T, 6),
            "intermediate": torch.randn(B, T, 6),
            "final": torch.randn(B, T, 6),
        }

        target_sgn = torch.randn(B, N)
        target_digits = torch.zeros(B, N, D, base)
        target_digits[:, :, :, 5] = 1.0
        target_ops = torch.zeros(B, depth, n_ops)
        target_ops[:, :, 0] = 1.0
        target_initial_values = torch.randn(B, N)
        target_final_exec = torch.randn(B)
        target_statistics = {
            "initial": torch.randn(B, 6),
            "intermediate": torch.randn(B, 6),
            "final": torch.randn(B, 6),
        }

        cfg = DAGTrainConfig()
        cfg.max_digits = D - 1
        cfg.max_decimal_places = 1
        cfg.base = base

        # Test case 1: Valid final_token_pos
        valid_final_token_pos = torch.tensor(
            [2, 4], dtype=torch.long
        )  # Within bounds [0, T-1]
        losses = compute_dag_structure_loss(
            pred_sign_logits,
            pred_digit_logits,
            pred_op_logits,
            pred_statistics,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            target_statistics,
            valid_final_token_pos,
            cfg,
            uncertainty_params=torch.zeros(6),
        )
        assert torch.isfinite(losses["total_loss"])

        # Test case 2: final_token_pos >= sequence_length should raise error
        invalid_final_token_pos = torch.tensor(
            [3, T], dtype=torch.long
        )  # T=5, so position 5 is invalid
        with pytest.raises(
            ValueError, match="Final token positions exceed sequence length"
        ):
            compute_dag_structure_loss(
                pred_sign_logits,
                pred_digit_logits,
                pred_op_logits,
                pred_statistics,
                target_sgn,
                target_digits,
                target_ops,
                target_initial_values,
                target_final_exec,
                target_statistics,
                invalid_final_token_pos,
                cfg,
                uncertainty_params=torch.zeros(6),
            )

    def test_extremely_long_expressions(self):
        """Test handling of expressions that might exceed block_size."""
        # This would be caught during data generation, but let's test the principle
        very_long_text = " + ".join(str(i) for i in range(100))  # Very long expression
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode_ordinary(very_long_text)

        # If this were to be processed, final_token_pos would be len(tokens) - 1
        expected_final_pos = len(tokens) - 1

        # In practice, this would be truncated by tokenize_texts
        small_block_size = 10
        tokenized = tokenize_texts([very_long_text], small_block_size, "cpu")

        # Should be truncated to block_size
        assert tokenized.shape == (1, small_block_size)
        # All positions should be non-padding (sequence was truncated)
        assert (tokenized[0] != 0).all()
