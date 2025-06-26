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

    def test_runpod_storage_complete_workflow(self, tmp_path, capsys):
        """Comprehensive test of RunPod storage: detection, saving, checking, and restoration."""
        # Setup directories
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # Test 1: RunPod detection
        with patch("data.common_prep.Path", return_value=runpod_volume):
            assert _get_runpod_storage_path() == runpod_volume

        # Test 2: Without RunPod - local only behavior
        prep_local = DataPrep(data_dir / "local_test")
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)

        with patch("data.common_prep._get_runpod_storage_path", return_value=None):
            prep_local.write_binary_file(test_data, "train")
            prep_local.save_meta()
            prep_local.print_completion("test-dataset", 1000, 200)

        # Verify local-only behavior
        assert (prep_local.data_dir / "train.bin").exists()
        assert (prep_local.data_dir / "meta.pkl").exists()
        local_data = np.fromfile(prep_local.data_dir / "train.bin", dtype=np.uint16)
        np.testing.assert_array_equal(local_data, test_data)

        # Test 3: With RunPod - dual storage and workflow
        prep_runpod = DataPrep(data_dir / "runpod_test", dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Initial preparation - saves to both locations
            prep_runpod.write_binary_file(test_data, "train")
            prep_runpod.write_binary_file(test_data[:3], "val")
            prep_runpod.save_meta(vocab_size=12345, tokenizer="custom")

            # Verify dual storage
            assert (prep_runpod.data_dir / "train.bin").exists()
            runpod_data_dir = runpod_volume / "data" / "shakespeare"
            assert (runpod_data_dir / "train.bin").exists()
            assert (runpod_data_dir / "val.bin").exists()
            assert (runpod_data_dir / "meta.pkl").exists()

            # Verify file contents match
            local_train = np.fromfile(
                prep_runpod.data_dir / "train.bin", dtype=np.uint16
            )
            runpod_train = np.fromfile(runpod_data_dir / "train.bin", dtype=np.uint16)
            np.testing.assert_array_equal(local_train, runpod_train)

            # Test completion message
            prep_runpod.print_completion("shakespeare", 5, 3)

        # Test 4: File existence checking and restoration workflow
        # Simulate pod restart by deleting local files
        shutil.rmtree(prep_runpod.data_dir)
        prep_restart = DataPrep(data_dir / "runpod_test", dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Should find files in RunPod and restore them
            assert prep_restart.check_existing_files() is True
            token_counts = prep_restart.get_existing_token_counts()
            assert token_counts == (5, 3)

            # Verify files were restored locally
            assert (prep_restart.data_dir / "train.bin").exists()
            assert (prep_restart.data_dir / "val.bin").exists()
            assert (prep_restart.data_dir / "meta.pkl").exists()

        # Test 5: Edge cases
        # Partial files in RunPod
        prep_partial = DataPrep(data_dir / "partial_test", dataset_name="partial")
        partial_runpod_dir = runpod_volume / "data" / "partial"
        partial_runpod_dir.mkdir(parents=True)
        (partial_runpod_dir / "train.bin").write_bytes(b"train data")
        # Missing val.bin and meta.pkl

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            assert prep_partial.check_existing_files() is False

        # Verify console output
        captured = capsys.readouterr()
        assert "📁 Copied train.bin to RunPod storage:" in captured.out
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "📥 Copied train.bin from RunPod storage to local" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out
        assert "📁 Files also saved to RunPod storage:" in captured.out

    def test_runpod_no_volume_detection(self, tmp_path):
        """Test RunPod detection when volume doesn't exist."""
        fake_runpod_volume = tmp_path / "runpod-volume"
        # Don't create the directory

        with patch("data.common_prep.Path", return_value=fake_runpod_volume):
            result = _get_runpod_storage_path()
            assert result is None


class TestDatasetOrganization:
    """Test dataset-specific folder organization and file existence checking."""

    def test_dataset_organization_and_caching(self, tmp_path, capsys):
        """Comprehensive test of dataset organization, file checking, and force parameter."""
        data_dir = tmp_path / "data"

        # Test 1: Dataset-specific folder creation
        prep_named = DataPrep(data_dir, dataset_name="shakespeare")
        expected_path = data_dir / "shakespeare"
        assert prep_named.data_dir == expected_path
        assert expected_path.exists()

        # Test 2: No dataset name - use original path
        prep_unnamed = DataPrep(data_dir)
        assert prep_unnamed.data_dir == data_dir

        # Test 3: File existence checking workflow
        # Create required files
        (prep_named.data_dir / "train.bin").write_bytes(b"train data")
        (prep_named.data_dir / "val.bin").write_bytes(b"val data")
        (prep_named.data_dir / "meta.pkl").write_bytes(b"meta data")

        assert prep_named.check_existing_files() is True

        # Test with only some files
        prep_partial = DataPrep(data_dir, dataset_name="partial")
        (prep_partial.data_dir / "train.bin").write_bytes(b"train data")
        assert prep_partial.check_existing_files() is False

        # Test 4: Custom file list checking
        prep_custom = DataPrep(data_dir, dataset_name="custom")
        (prep_custom.data_dir / "custom.bin").write_bytes(b"custom data")
        (prep_custom.data_dir / "special.pkl").write_bytes(b"special data")

        assert prep_custom.check_existing_files(["custom.bin", "special.pkl"]) is True
        assert prep_custom.check_existing_files(["custom.bin", "missing.pkl"]) is False

        # Test 5: Token count reading
        train_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        val_data = np.array([6, 7, 8], dtype=np.uint16)

        prep_tokens = DataPrep(data_dir, dataset_name="tokens")
        train_data.tofile(prep_tokens.data_dir / "train.bin")
        val_data.tofile(prep_tokens.data_dir / "val.bin")

        # Create meta file
        meta = {"vocab_size": 50257, "tokenizer": "gpt2"}
        with (prep_tokens.data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        token_counts = prep_tokens.get_existing_token_counts()
        assert token_counts == (5, 3)

        # Test 6: Byte-level encoding
        prep_bytes = DataPrep(data_dir, dataset_name="bytes")
        byte_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
        byte_data.tofile(prep_bytes.data_dir / "train.bin")
        byte_data[:3].tofile(prep_bytes.data_dir / "val.bin")

        meta_bytes = {"vocab_size": 256, "tokenizer": "byte", "byte_level": True}
        with (prep_bytes.data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta_bytes, f)

        token_counts_bytes = prep_bytes.get_existing_token_counts()
        assert token_counts_bytes == (8, 3)

        # Test 7: Missing files
        prep_missing = DataPrep(data_dir, dataset_name="missing")
        assert prep_missing.get_existing_token_counts() is None

        # Verify console output
        captured = capsys.readouterr()
        assert "✅ All required files found locally" in captured.out


class TestForceParameter:
    """Test the force parameter functionality."""

    def test_force_parameter_and_dataset_separation(self, tmp_path, capsys):
        """Test force parameter in parser and dataset separation in RunPod storage."""
        # Test 1: Force parameter in parser
        from data.common_prep import get_common_parser

        parser = get_common_parser("Test parser")

        # Parse arguments with force flag
        args = parser.parse_args(["--force"])
        assert args.force is True

        # Parse without force flag
        args = parser.parse_args([])
        assert args.force is False

        # Test 2: Dataset-specific RunPod storage separation
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


class TestRunPodIntegration:
    """Test complete RunPod integration scenarios."""

    def test_complete_preparation_workflow(self, tmp_path, capsys):
        """Test complete workflow: prepare → restart → restore → use cache."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # Scenario 1: First preparation
        prep1 = DataPrep(data_dir, dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Simulate dataset preparation
            test_text = "To be or not to be, that is the question."
            train_ids = prep1.tokenize_text(test_text, add_eot=False)
            val_ids = prep1.tokenize_text(test_text[:10], add_eot=False)

            train_array = np.array(train_ids, dtype=np.uint16)
            val_array = np.array(val_ids, dtype=np.uint16)

            prep1.write_binary_file(train_array, "train")
            prep1.write_binary_file(val_array, "val")
            prep1.save_meta()

        # Verify files exist in both locations
        assert (prep1.data_dir / "train.bin").exists()
        runpod_data_dir = runpod_volume / "data" / "shakespeare"
        assert (runpod_data_dir / "train.bin").exists()

        # Scenario 2: Simulate pod restart (delete local files)
        shutil.rmtree(prep1.data_dir)

        # Scenario 3: Restoration from RunPod
        prep2 = DataPrep(data_dir, dataset_name="shakespeare")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            files_exist = prep2.check_existing_files()
            assert files_exist is True

            token_counts = prep2.get_existing_token_counts()
            assert token_counts == (len(train_array), len(val_array))

        # Scenario 4: Local cache usage (subsequent runs)
        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            prep3 = DataPrep(data_dir, dataset_name="shakespeare")
            assert prep3.check_existing_files() is True  # Uses local cache

        # Verify console messages
        captured = capsys.readouterr()
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out
        assert "✅ All required files found locally" in captured.out

    def test_copy_from_runpod_error_handling(self, tmp_path, capsys):
        """Test error handling when copying from RunPod storage."""
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
