#!/usr/bin/env python
"""Test the common_prep module functionality."""

import pickle
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from data.common_prep import DataPrep, _get_runpod_storage_path


class TestRunPodStorage:
    """Test RunPod persistent storage functionality."""

    def test_get_runpod_storage_path_exists(self, tmp_path):
        """Test RunPod storage path detection when volume exists."""
        fake_runpod_volume = tmp_path / "runpod-volume"
        fake_runpod_volume.mkdir()

        # Mock the Path constructor to return our fake volume
        with patch("data.common_prep.Path", return_value=fake_runpod_volume):
            result = _get_runpod_storage_path()
            assert result == fake_runpod_volume

    def test_get_runpod_storage_path_not_exists(self, tmp_path):
        """Test RunPod storage path detection when volume doesn't exist."""
        fake_runpod_volume = tmp_path / "runpod-volume"
        # Don't create the directory, so exists() returns False

        with patch("data.common_prep.Path", return_value=fake_runpod_volume):
            result = _get_runpod_storage_path()
            assert result is None

    def test_local_behavior_without_runpod(self, tmp_path):
        """Test that local behavior is preserved when RunPod volume doesn't exist."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir)

        # Create test data
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)

        with patch("data.common_prep._get_runpod_storage_path", return_value=None):
            prep.write_binary_file(test_data, "train")
            prep.save_meta()

        # Verify files exist locally
        assert (data_dir / "train.bin").exists()
        assert (data_dir / "meta.pkl").exists()

        # Verify file contents
        loaded_data = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
        np.testing.assert_array_equal(loaded_data, test_data)

        with (data_dir / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        assert meta["vocab_size"] == 50257
        assert meta["tokenizer"] == "gpt2"

    def test_runpod_storage_copy_behavior(self, tmp_path, capsys):
        """Test that files are copied to RunPod storage when available."""
        # Setup directories
        data_dir = tmp_path / "local_data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir)

        # Create test data
        test_data = np.array([10, 20, 30, 40, 50], dtype=np.uint16)

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            prep.write_binary_file(test_data, "val")
            prep.save_meta(vocab_size=12345, tokenizer="custom")

        # Verify local files exist
        assert (data_dir / "val.bin").exists()
        assert (data_dir / "meta.pkl").exists()

        # Verify RunPod storage files exist
        runpod_data_dir = runpod_volume / "data" / data_dir.name
        assert (runpod_data_dir / "val.bin").exists()
        assert (runpod_data_dir / "meta.pkl").exists()

        # Verify file contents in both locations
        local_data = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
        runpod_data = np.fromfile(runpod_data_dir / "val.bin", dtype=np.uint16)
        np.testing.assert_array_equal(local_data, test_data)
        np.testing.assert_array_equal(runpod_data, test_data)

        # Verify metadata in both locations
        with (data_dir / "meta.pkl").open("rb") as f:
            local_meta = pickle.load(f)
        with (runpod_data_dir / "meta.pkl").open("rb") as f:
            runpod_meta = pickle.load(f)

        assert local_meta == runpod_meta
        assert local_meta["vocab_size"] == 12345
        assert local_meta["tokenizer"] == "custom"

        # Verify console output shows copy messages
        captured = capsys.readouterr()
        assert "📁 Copied val.bin to RunPod storage:" in captured.out
        assert "📁 Copied meta.pkl to RunPod storage:" in captured.out

    def test_print_completion_with_runpod(self, tmp_path, capsys):
        """Test completion message includes RunPod storage info when available."""
        data_dir = tmp_path / "test_data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir)

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            prep.print_completion("test-dataset", 1000, 200)

        captured = capsys.readouterr()
        assert "✅ Preparation complete for test-dataset" in captured.out
        assert "Train tokens: 1,000" in captured.out
        assert "Val tokens:   200" in captured.out
        assert "📁 Files also saved to RunPod storage:" in captured.out
        assert str(runpod_volume / "data" / data_dir.name) in captured.out

    def test_print_completion_without_runpod(self, tmp_path, capsys):
        """Test completion message without RunPod storage info when not available."""
        data_dir = tmp_path / "test_data"
        prep = DataPrep(data_dir)

        with patch("data.common_prep._get_runpod_storage_path", return_value=None):
            prep.print_completion("test-dataset", 1000, 200)

        captured = capsys.readouterr()
        assert "✅ Preparation complete for test-dataset" in captured.out
        assert "Train tokens: 1,000" in captured.out
        assert "Val tokens:   200" in captured.out
        assert "📁 Files also saved to RunPod storage:" not in captured.out

    def test_dataset_tokenization_with_runpod(self, tmp_path):
        """Test dataset tokenization with RunPod storage using mock dataset."""
        data_dir = tmp_path / "dataset_test"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir)

        # Create mock dataset-like object
        class MockDataset:
            def __init__(self, token_ids, lengths):
                self.token_ids = token_ids
                self.lengths = lengths
                self.size = len(token_ids)

            def __len__(self):
                return self.size

            def shard(self, num_shards, shard_idx, contiguous=True):
                return self

            def with_format(self, format_type):
                return {"ids": self.token_ids, "len": self.lengths}

            def __getitem__(self, key):
                if key == "len":
                    return self.lengths
                return getattr(self, key)

        # Create test dataset
        test_ids = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        test_lengths = [3, 2, 4]
        mock_dataset = MockDataset(test_ids, test_lengths)

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            with patch("numpy.sum", return_value=9):  # Total tokens
                prep.write_binary_file(mock_dataset, "train")

        # Verify files exist in both locations
        assert (data_dir / "train.bin").exists()
        runpod_data_dir = runpod_volume / "data" / data_dir.name
        assert (runpod_data_dir / "train.bin").exists()

    def test_copy_to_runpod_storage_creates_directories(self, tmp_path):
        """Test that RunPod storage directory structure is created properly."""
        data_dir = tmp_path / "nested" / "data" / "dir"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir)

        # Create a test file
        test_file = data_dir / "test.bin"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test data")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            prep._copy_to_runpod_storage(test_file)

        # Verify directory structure was created
        expected_runpod_dir = runpod_volume / "data" / data_dir.name
        assert expected_runpod_dir.exists()
        assert (expected_runpod_dir / "test.bin").exists()
        assert (expected_runpod_dir / "test.bin").read_bytes() == b"test data"

    def test_integration_shakespeare_with_runpod(self, tmp_path, capsys):
        """Integration test demonstrating RunPod storage with Shakespeare-style data preparation."""
        import os
        import tempfile

        # Setup directories
        data_dir = tmp_path / "shakespeare_test"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir)

        # Create some test text data (like Shakespeare would have)
        test_text = "To be or not to be, that is the question. " * 100

        # Simulate the Shakespeare preparation process
        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Tokenize the text
            train_ids = prep.tokenize_text(test_text, add_eot=False)
            val_ids = prep.tokenize_text(
                test_text[:100], add_eot=False
            )  # Smaller val set

            # Convert to numpy arrays (like Shakespeare script does)
            train_ids = np.array(train_ids, dtype=np.uint16)
            val_ids = np.array(val_ids, dtype=np.uint16)

            # Write binary files
            prep.write_binary_file(train_ids, "train")
            prep.write_binary_file(val_ids, "val")

            # Save metadata
            prep.save_meta()

            # Print completion
            prep.print_completion("shakespeare", len(train_ids), len(val_ids))

        # Verify local files exist
        assert (data_dir / "train.bin").exists()
        assert (data_dir / "val.bin").exists()
        assert (data_dir / "meta.pkl").exists()

        # Verify RunPod storage files exist
        runpod_data_dir = runpod_volume / "data" / data_dir.name
        assert (runpod_data_dir / "train.bin").exists()
        assert (runpod_data_dir / "val.bin").exists()
        assert (runpod_data_dir / "meta.pkl").exists()

        # Verify file contents match
        local_train = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
        runpod_train = np.fromfile(runpod_data_dir / "train.bin", dtype=np.uint16)
        np.testing.assert_array_equal(local_train, runpod_train)

        local_val = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
        runpod_val = np.fromfile(runpod_data_dir / "val.bin", dtype=np.uint16)
        np.testing.assert_array_equal(local_val, runpod_val)

        # Verify console output shows RunPod storage messages
        captured = capsys.readouterr()
        assert "📁 Copied train.bin to RunPod storage:" in captured.out
        assert "📁 Copied val.bin to RunPod storage:" in captured.out
        assert "📁 Copied meta.pkl to RunPod storage:" in captured.out
        assert "📁 Files also saved to RunPod storage:" in captured.out
        assert "✅ Preparation complete for shakespeare" in captured.out


class TestDatasetSubfolders:
    """Test dataset-specific subfolder organization and file existence checking."""

    def test_dataset_subfolder_creation(self, tmp_path):
        """Test that dataset-specific subfolders are created correctly."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="shakespeare")

        # Should create data/shakespeare subfolder
        expected_path = data_dir / "shakespeare"
        assert prep.data_dir == expected_path
        assert expected_path.exists()

    def test_dataset_subfolder_without_name(self, tmp_path):
        """Test that no subfolder is created when dataset_name is None."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir)

        # Should use the original data_dir
        assert prep.data_dir == data_dir
        assert data_dir.exists()

    def test_check_existing_files_all_present(self, tmp_path):
        """Test file existence check when all required files are present."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create required files
        (prep.data_dir / "train.bin").write_bytes(b"train data")
        (prep.data_dir / "val.bin").write_bytes(b"val data")
        (prep.data_dir / "meta.pkl").write_bytes(b"meta data")

        assert prep.check_existing_files() is True

    def test_check_existing_files_missing_some(self, tmp_path):
        """Test file existence check when some files are missing."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create only some files
        (prep.data_dir / "train.bin").write_bytes(b"train data")
        # val.bin and meta.pkl missing

        assert prep.check_existing_files() is False

    def test_check_existing_files_custom_list(self, tmp_path):
        """Test file existence check with custom file list."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create custom files
        (prep.data_dir / "custom.bin").write_bytes(b"custom data")
        (prep.data_dir / "special.pkl").write_bytes(b"special data")

        assert prep.check_existing_files(["custom.bin", "special.pkl"]) is True
        assert prep.check_existing_files(["custom.bin", "missing.pkl"]) is False

    def test_get_existing_token_counts_success(self, tmp_path):
        """Test getting token counts from existing files."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create test data
        train_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        val_data = np.array([6, 7, 8], dtype=np.uint16)

        train_data.tofile(prep.data_dir / "train.bin")
        val_data.tofile(prep.data_dir / "val.bin")

        # Create meta file
        meta = {"vocab_size": 50257, "tokenizer": "gpt2"}
        with (prep.data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        token_counts = prep.get_existing_token_counts()
        assert token_counts == (5, 3)

    def test_get_existing_token_counts_missing_files(self, tmp_path):
        """Test getting token counts when files are missing."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        token_counts = prep.get_existing_token_counts()
        assert token_counts is None

    def test_get_existing_token_counts_byte_level(self, tmp_path):
        """Test getting token counts with byte-level encoding."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create test data with uint8 dtype (byte-level)
        train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
        val_data = np.array([9, 10, 11], dtype=np.uint8)

        train_data.tofile(prep.data_dir / "train.bin")
        val_data.tofile(prep.data_dir / "val.bin")

        # Create meta file with byte_level flag
        meta = {"vocab_size": 256, "tokenizer": "byte", "byte_level": True}
        with (prep.data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        token_counts = prep.get_existing_token_counts()
        assert token_counts == (8, 3)


class TestForceParameter:
    """Test the force parameter functionality."""

    def test_force_parameter_in_parser(self):
        """Test that the --force flag is added to the common parser."""
        from data.common_prep import get_common_parser

        parser = get_common_parser("Test parser")

        # Parse arguments with force flag
        args = parser.parse_args(["--force"])
        assert args.force is True

        # Parse without force flag
        args = parser.parse_args([])
        assert args.force is False

    def test_dataset_specific_runpod_storage(self, tmp_path, capsys):
        """Test that RunPod storage uses dataset-specific paths."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # Test different datasets create separate folders
        datasets = ["shakespeare", "openwebtext", "proofpile"]

        for dataset in datasets:
            prep = DataPrep(data_dir, dataset_name=dataset)
            test_data = np.array([1, 2, 3], dtype=np.uint16)

            with patch(
                "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
            ):
                prep.write_binary_file(test_data, "train")

            # Verify dataset-specific folder was created in RunPod storage
            expected_runpod_path = runpod_volume / "data" / dataset
            assert expected_runpod_path.exists()
            assert (expected_runpod_path / "train.bin").exists()

        # Verify all datasets have separate folders
        for dataset in datasets:
            assert (runpod_volume / "data" / dataset).exists()

        # Verify console output shows dataset-specific paths
        captured = capsys.readouterr()
        for dataset in datasets:
            assert f"runpod-volume/data/{dataset}/train.bin" in captured.out


class TestRunPodPreparationChecking:
    """Test RunPod storage checking before preparation."""

    def test_check_existing_files_local_only(self, tmp_path, capsys):
        """Test file checking when files exist locally but no RunPod."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # Create local files
        (prep.data_dir / "train.bin").write_bytes(b"train data")
        (prep.data_dir / "val.bin").write_bytes(b"val data")
        (prep.data_dir / "meta.pkl").write_bytes(b"meta data")

        # No RunPod available
        with patch("data.common_prep._get_runpod_storage_path", return_value=None):
            assert prep.check_existing_files() is True

        captured = capsys.readouterr()
        assert "✅ All required files found locally" in captured.out

    def test_check_existing_files_missing_local_no_runpod(self, tmp_path, capsys):
        """Test file checking when files missing locally and no RunPod."""
        data_dir = tmp_path / "data"
        prep = DataPrep(data_dir, dataset_name="test")

        # No local files, no RunPod
        with patch("data.common_prep._get_runpod_storage_path", return_value=None):
            assert prep.check_existing_files() is False

        captured = capsys.readouterr()
        assert "Missing files:" in captured.out

    def test_check_existing_files_found_in_runpod(self, tmp_path, capsys):
        """Test file checking when files missing locally but found in RunPod."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir, dataset_name="test")

        # Create files in RunPod storage
        runpod_data_dir = runpod_volume / "data" / "test"
        runpod_data_dir.mkdir(parents=True)
        (runpod_data_dir / "train.bin").write_bytes(b"train data")
        (runpod_data_dir / "val.bin").write_bytes(b"val data")
        (runpod_data_dir / "meta.pkl").write_bytes(b"meta data")

        # No local files initially
        assert not (prep.data_dir / "train.bin").exists()

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            assert prep.check_existing_files() is True

        # Verify files were copied locally
        assert (prep.data_dir / "train.bin").exists()
        assert (prep.data_dir / "val.bin").exists()
        assert (prep.data_dir / "meta.pkl").exists()

        captured = capsys.readouterr()
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "📥 Copied train.bin from RunPod storage to local" in captured.out
        assert "📥 Copied val.bin from RunPod storage to local" in captured.out
        assert "📥 Copied meta.pkl from RunPod storage to local" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out

    def test_check_existing_files_partial_in_runpod(self, tmp_path, capsys):
        """Test file checking when only some files exist in RunPod."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir, dataset_name="test")

        # Create only some files in RunPod storage
        runpod_data_dir = runpod_volume / "data" / "test"
        runpod_data_dir.mkdir(parents=True)
        (runpod_data_dir / "train.bin").write_bytes(b"train data")
        # Missing val.bin and meta.pkl

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            assert prep.check_existing_files() is False

        captured = capsys.readouterr()
        assert "Missing files (local and RunPod):" in captured.out

    def test_get_existing_token_counts_from_runpod(self, tmp_path):
        """Test getting token counts when files are only in RunPod storage."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir, dataset_name="test")

        # Create test data in RunPod storage only
        runpod_data_dir = runpod_volume / "data" / "test"
        runpod_data_dir.mkdir(parents=True)

        train_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        val_data = np.array([6, 7, 8], dtype=np.uint16)

        train_data.tofile(runpod_data_dir / "train.bin")
        val_data.tofile(runpod_data_dir / "val.bin")

        # Create meta file in RunPod
        meta = {"vocab_size": 50257, "tokenizer": "gpt2"}
        with (runpod_data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        # No local files initially
        assert not (prep.data_dir / "train.bin").exists()

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            token_counts = prep.get_existing_token_counts()

        assert token_counts == (5, 3)

        # Verify files were copied locally
        assert (prep.data_dir / "train.bin").exists()
        assert (prep.data_dir / "val.bin").exists()
        assert (prep.data_dir / "meta.pkl").exists()

    def test_copy_from_runpod_storage_warning(self, tmp_path, capsys):
        """Test warning when trying to copy non-existent files from RunPod."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        prep = DataPrep(data_dir, dataset_name="test")
        runpod_data_dir = runpod_volume / "data" / "test"
        runpod_data_dir.mkdir(parents=True)

        # Try to copy non-existent files
        prep._copy_from_runpod_storage(runpod_data_dir, ["missing.bin"])

        captured = capsys.readouterr()
        assert "⚠️ Warning: missing.bin not found in RunPod storage" in captured.out

    def test_integration_preparation_with_runpod_cache(self, tmp_path, capsys):
        """Integration test: prepare once, then use RunPod cache on second run."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # First preparation - create and save to RunPod
        prep1 = DataPrep(data_dir, dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Simulate preparation
            test_text = "To be or not to be, that is the question."
            train_ids = prep1.tokenize_text(test_text, add_eot=False)
            val_ids = prep1.tokenize_text(test_text[:10], add_eot=False)

            train_ids = np.array(train_ids, dtype=np.uint16)
            val_ids = np.array(val_ids, dtype=np.uint16)

            prep1.write_binary_file(train_ids, "train")
            prep1.write_binary_file(val_ids, "val")
            prep1.save_meta()

        # Verify files exist in both local and RunPod
        assert (prep1.data_dir / "train.bin").exists()
        runpod_data_dir = runpod_volume / "data" / "shakespeare"
        assert (runpod_data_dir / "train.bin").exists()

        # Delete local files to simulate fresh environment
        import shutil

        shutil.rmtree(prep1.data_dir)

        # Second preparation attempt - should use RunPod cache
        prep2 = DataPrep(data_dir, dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            files_exist = prep2.check_existing_files()
            assert files_exist is True

            token_counts = prep2.get_existing_token_counts()
            assert token_counts == (len(train_ids), len(val_ids))

        # Verify files were restored locally
        assert (prep2.data_dir / "train.bin").exists()
        assert (prep2.data_dir / "val.bin").exists()
        assert (prep2.data_dir / "meta.pkl").exists()

        captured = capsys.readouterr()
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out
