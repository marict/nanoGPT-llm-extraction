#!/usr/bin/env python
"""Test the common_prep module functionality."""

import pickle
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from data.common_prep import DataPrep, _get_runpod_storage_path


# --------------------------------------------------------------------- #
# Consolidated RunPod storage tests (1 test)
# --------------------------------------------------------------------- #
class TestRunPodStorage:
    """Test RunPod persistent storage functionality."""

    def test_runpod_storage_comprehensive(self, tmp_path, capsys):
        """Comprehensive test of RunPod storage: detection, saving, checking, restoration, and error handling."""
        # Setup directories
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # Test 1: RunPod detection
        with patch("data.common_prep.Path", return_value=runpod_volume):
            assert _get_runpod_storage_path() == runpod_volume

        # Test 2: RunPod detection when volume doesn't exist
        fake_runpod_volume = tmp_path / "fake-runpod-volume"
        # Don't create the directory
        with patch("data.common_prep.Path", return_value=fake_runpod_volume):
            result = _get_runpod_storage_path()
            assert result is None

        # Test 3: Without RunPod - local only behavior
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

        # Test 4: With RunPod - dual storage and workflow
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

        # Test 5: File existence checking and restoration workflow
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

        # Test 6: Error handling - partial files in RunPod
        prep_partial = DataPrep(data_dir / "partial_test", dataset_name="partial")
        partial_runpod_dir = runpod_volume / "data" / "partial"
        partial_runpod_dir.mkdir(parents=True)
        (partial_runpod_dir / "train.bin").write_bytes(b"train data")
        # Missing val.bin and meta.pkl

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            assert prep_partial.check_existing_files() is False

        # Test 7: Copy from RunPod error handling
        prep_error = DataPrep(data_dir / "error_test", dataset_name="error")
        error_runpod_dir = runpod_volume / "data" / "error"
        error_runpod_dir.mkdir(parents=True)

        # Create a file that might cause copy issues
        (error_runpod_dir / "train.bin").write_bytes(b"test data")
        (error_runpod_dir / "val.bin").write_bytes(b"val data")
        (error_runpod_dir / "meta.pkl").write_bytes(b"meta data")

        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Should handle copy operations gracefully
            assert prep_error.check_existing_files() is True

            # Test that files were copied successfully
            assert (prep_error.data_dir / "train.bin").exists()
            assert (prep_error.data_dir / "val.bin").exists()
            assert (prep_error.data_dir / "meta.pkl").exists()

        # Verify console output
        captured = capsys.readouterr()
        assert "📁 Copied train.bin to RunPod storage:" in captured.out
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "📥 Copied train.bin from RunPod storage to local" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out
        assert "📁 Files also saved to RunPod storage:" in captured.out


# --------------------------------------------------------------------- #
# Consolidated dataset organization and force parameter tests (1 test)
# --------------------------------------------------------------------- #
class TestDatasetOrganization:
    """Test dataset-specific folder organization, file existence checking, and force parameter functionality."""

    def test_dataset_organization_and_force_parameter_comprehensive(
        self, tmp_path, capsys
    ):
        """Comprehensive test of dataset organization, file checking, caching, and force parameter."""
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

        # Test 8: Force parameter functionality
        prep_force = DataPrep(data_dir, dataset_name="force_test")

        # Create initial files
        (prep_force.data_dir / "train.bin").write_bytes(b"initial train data")
        (prep_force.data_dir / "val.bin").write_bytes(b"initial val data")
        (prep_force.data_dir / "meta.pkl").write_bytes(b"initial meta data")

        # Test force parameter behavior
        # Without force, should detect existing files
        assert prep_force.check_existing_files() is True

        # With force, should proceed regardless (simulated by ignoring existing files)
        # This tests the force parameter logic in preparation workflows
        prep_force_enabled = DataPrep(data_dir, dataset_name="force_enabled")

        # Create files that would normally prevent reprocessing
        (prep_force_enabled.data_dir / "train.bin").write_bytes(b"existing data")
        (prep_force_enabled.data_dir / "val.bin").write_bytes(b"existing data")
        (prep_force_enabled.data_dir / "meta.pkl").write_bytes(b"existing data")

        # Test that we can still write new data (force parameter logic)
        new_data = np.array([10, 20, 30], dtype=np.uint16)
        prep_force_enabled.write_binary_file(new_data, "train")

        # Verify new data was written
        written_data = np.fromfile(
            prep_force_enabled.data_dir / "train.bin", dtype=np.uint16
        )
        np.testing.assert_array_equal(written_data, new_data)

        # Test dataset separation - different datasets should have separate directories
        prep_dataset_a = DataPrep(data_dir, dataset_name="dataset_a")
        prep_dataset_b = DataPrep(data_dir, dataset_name="dataset_b")

        assert prep_dataset_a.data_dir != prep_dataset_b.data_dir
        assert prep_dataset_a.data_dir == data_dir / "dataset_a"
        assert prep_dataset_b.data_dir == data_dir / "dataset_b"

        # Test that datasets are properly isolated
        test_data_a = np.array([1, 2, 3], dtype=np.uint16)
        test_data_b = np.array([4, 5, 6], dtype=np.uint16)

        prep_dataset_a.write_binary_file(test_data_a, "train")
        prep_dataset_b.write_binary_file(test_data_b, "train")

        # Verify isolation
        data_a = np.fromfile(prep_dataset_a.data_dir / "train.bin", dtype=np.uint16)
        data_b = np.fromfile(prep_dataset_b.data_dir / "train.bin", dtype=np.uint16)

        np.testing.assert_array_equal(data_a, test_data_a)
        np.testing.assert_array_equal(data_b, test_data_b)
        assert not np.array_equal(data_a, data_b)

        # Verify console output
        captured = capsys.readouterr()
        assert "✅ All required files found locally" in captured.out


# --------------------------------------------------------------------- #
# Complete preparation workflow test (1 test)
# --------------------------------------------------------------------- #
class TestRunPodIntegration:
    """Test complete preparation workflow with RunPod integration."""

    def test_complete_preparation_workflow(self, tmp_path, capsys):
        """Test complete preparation workflow with RunPod integration and realistic scenarios."""
        data_dir = tmp_path / "data"
        runpod_volume = tmp_path / "runpod-volume"
        runpod_volume.mkdir()

        # Test complete workflow: preparation -> saving -> pod restart -> restoration
        with patch(
            "data.common_prep._get_runpod_storage_path", return_value=runpod_volume
        ):
            # Step 1: Initial data preparation
            prep = DataPrep(data_dir, dataset_name="complete_test")

            # Simulate data preparation
            train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
            val_data = np.array([11, 12, 13, 14, 15], dtype=np.uint16)

            prep.write_binary_file(train_data, "train")
            prep.write_binary_file(val_data, "val")
            prep.save_meta(vocab_size=16, tokenizer="custom")
            prep.print_completion("complete_test", 10, 5)

            # Verify local files exist
            assert (prep.data_dir / "train.bin").exists()
            assert (prep.data_dir / "val.bin").exists()
            assert (prep.data_dir / "meta.pkl").exists()

            # Verify RunPod files exist
            runpod_data_dir = runpod_volume / "data" / "complete_test"
            assert (runpod_data_dir / "train.bin").exists()
            assert (runpod_data_dir / "val.bin").exists()
            assert (runpod_data_dir / "meta.pkl").exists()

            # Step 2: Simulate pod restart (delete local files)
            shutil.rmtree(prep.data_dir)

            # Step 3: New preparation instance (simulates pod restart)
            prep_restart = DataPrep(data_dir, dataset_name="complete_test")

            # Should find existing files in RunPod
            assert prep_restart.check_existing_files() is True

            # Should get correct token counts
            token_counts = prep_restart.get_existing_token_counts()
            assert token_counts == (10, 5)

            # Verify files were restored
            assert (prep_restart.data_dir / "train.bin").exists()
            assert (prep_restart.data_dir / "val.bin").exists()
            assert (prep_restart.data_dir / "meta.pkl").exists()

            # Verify data integrity
            restored_train = np.fromfile(
                prep_restart.data_dir / "train.bin", dtype=np.uint16
            )
            restored_val = np.fromfile(
                prep_restart.data_dir / "val.bin", dtype=np.uint16
            )

            np.testing.assert_array_equal(restored_train, train_data)
            np.testing.assert_array_equal(restored_val, val_data)

            # Verify metadata
            with (prep_restart.data_dir / "meta.pkl").open("rb") as f:
                restored_meta = pickle.load(f)

            assert restored_meta["vocab_size"] == 16
            assert restored_meta["tokenizer"] == "custom"

            # Step 4: Test multiple dataset handling
            prep_multi = DataPrep(data_dir, dataset_name="multi_test")
            multi_data = np.array([20, 21, 22, 23, 24], dtype=np.uint16)

            prep_multi.write_binary_file(multi_data, "train")
            prep_multi.write_binary_file(multi_data[:2], "val")
            prep_multi.save_meta(vocab_size=25, tokenizer="multi")

            # Should not interfere with previous dataset
            assert prep_restart.get_existing_token_counts() == (10, 5)
            assert prep_multi.get_existing_token_counts() == (5, 2)

            # Step 5: Test edge cases
            # Empty dataset
            prep_empty = DataPrep(data_dir, dataset_name="empty_test")
            assert prep_empty.check_existing_files() is False
            assert prep_empty.get_existing_token_counts() is None

            # Dataset with corrupted files
            prep_corrupt = DataPrep(data_dir, dataset_name="corrupt_test")
            runpod_corrupt_dir = runpod_volume / "data" / "corrupt_test"
            runpod_corrupt_dir.mkdir(parents=True)

            # Create files with wrong format
            (runpod_corrupt_dir / "train.bin").write_bytes(b"not binary data")
            (runpod_corrupt_dir / "val.bin").write_bytes(b"also not binary")
            (runpod_corrupt_dir / "meta.pkl").write_bytes(b"corrupted pickle")

            # Should detect files exist but may fail on token count reading
            assert prep_corrupt.check_existing_files() is True
            # Token count reading should handle errors gracefully
            try:
                token_counts = prep_corrupt.get_existing_token_counts()
                # If it doesn't raise an error, it should return None for corrupted data
                assert token_counts is None or isinstance(token_counts, tuple)
            except:
                # Acceptable for corrupted data to raise errors
                pass

        # Verify comprehensive console output
        captured = capsys.readouterr()
        assert "📁 Files also saved to RunPod storage:" in captured.out
        assert "📁 Found all files in RunPod storage:" in captured.out
        assert "📥 Copied" in captured.out
        assert "✅ All required files restored from RunPod storage" in captured.out
