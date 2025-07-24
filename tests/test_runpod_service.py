"""Tests for runpod_service.py functionality."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runpod_service import (
    RunPodError,
    _create_docker_script,
    _validate_commit_hash,
    _validate_note,
    start_cloud_training,
)


class TestRunPodServiceCommitValidation(unittest.TestCase):
    """Test commit hash validation functionality."""

    def test_validate_commit_hash_valid_short(self):
        """Test validation with valid short commit hash."""
        # Mock git rev-parse to return a full hash
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45\n", returncode=0
            )
            result = _validate_commit_hash("4cc6ddb")
            self.assertEqual(result, "4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45")

    def test_validate_commit_hash_valid_full(self):
        """Test validation with valid full commit hash."""
        full_hash = "4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=f"{full_hash}\n", returncode=0)
            result = _validate_commit_hash(full_hash)
            self.assertEqual(result, full_hash)

    def test_validate_commit_hash_empty(self):
        """Test validation with empty commit hash returns as-is."""
        result = _validate_commit_hash("")
        self.assertEqual(result, "")

    def test_validate_commit_hash_none(self):
        """Test validation with None commit hash returns as-is."""
        result = _validate_commit_hash(None)
        self.assertIsNone(result)

    def test_validate_commit_hash_invalid_format(self):
        """Test validation with invalid format raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _validate_commit_hash("invalid_hash")
        self.assertIn("Invalid commit hash format", str(cm.exception))

    def test_validate_commit_hash_too_short(self):
        """Test validation with too short hash raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _validate_commit_hash("abc123")  # Only 6 characters
        self.assertIn("Invalid commit hash format", str(cm.exception))

    def test_validate_commit_hash_too_long(self):
        """Test validation with too long hash raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _validate_commit_hash("a" * 41)  # 41 characters
        self.assertIn("Invalid commit hash format", str(cm.exception))

    def test_validate_commit_hash_nonexistent(self):
        """Test validation with non-existent commit raises ValueError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            with self.assertRaises(ValueError) as cm:
                _validate_commit_hash("1234567")
            self.assertIn("Commit hash not found in repository", str(cm.exception))

    def test_validate_commit_hash_git_not_found(self):
        """Test validation when git command is not found raises RunPodError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with self.assertRaises(RunPodError) as cm:
                _validate_commit_hash("1234567")
            self.assertIn("Git command not found", str(cm.exception))


class TestDockerScriptCreation(unittest.TestCase):
    """Test Docker script creation functionality."""

    def test_create_docker_script_without_commit(self):
        """Test Docker script creation without commit hash."""
        command = "python train.py config.py"
        script = _create_docker_script(command)

        expected_commands = [
            "apt-get update && apt-get install -y git",
            "cd /workspace",
            "( [ -d repo/.git ] && git -C repo pull || git clone https://github.com/marict/nanoGPT-llm-extraction.git repo )",
            'cd /workspace/repo && echo "Repository version: $(git rev-parse HEAD)"',
            "bash /workspace/repo/scripts/container_setup.sh python train.py config.py",
        ]
        expected = " && ".join(expected_commands)

        self.assertEqual(script, expected)

    def test_create_docker_script_with_commit(self):
        """Test Docker script creation with commit hash."""
        command = "python train.py config.py"
        commit_hash = "4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45"
        script = _create_docker_script(command, commit_hash)

        expected_commands = [
            "apt-get update && apt-get install -y git",
            "cd /workspace",
            "( [ -d repo/.git ] && git -C repo pull || git clone https://github.com/marict/nanoGPT-llm-extraction.git repo )",
            f"cd /workspace/repo && git checkout {commit_hash}",
            'cd /workspace/repo && echo "Repository version: $(git rev-parse HEAD)"',
            "bash /workspace/repo/scripts/container_setup.sh python train.py config.py",
        ]
        expected = " && ".join(expected_commands)

        self.assertEqual(script, expected)

    def test_create_docker_script_with_none_commit(self):
        """Test Docker script creation with None commit hash."""
        command = "python train.py config.py"
        script = _create_docker_script(command, None)

        # Should be same as without commit
        expected_commands = [
            "apt-get update && apt-get install -y git",
            "cd /workspace",
            "( [ -d repo/.git ] && git -C repo pull || git clone https://github.com/marict/nanoGPT-llm-extraction.git repo )",
            'cd /workspace/repo && echo "Repository version: $(git rev-parse HEAD)"',
            "bash /workspace/repo/scripts/container_setup.sh python train.py config.py",
        ]
        expected = " && ".join(expected_commands)

        self.assertEqual(script, expected)


class TestStartCloudTrainingCommit(unittest.TestCase):
    """Test start_cloud_training with commit functionality."""

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    @patch("runpod_service.runpod")
    @patch("runpod_service.wandb")
    @patch("runpod_service._validate_commit_hash")
    @patch("runpod_service._extract_config_name")
    @patch("runpod_service._resolve_gpu_id")
    @patch("runpod_service.init_local_wandb_and_open_browser")
    def test_start_cloud_training_with_valid_commit(
        self,
        mock_wandb_init,
        mock_resolve_gpu,
        mock_extract_name,
        mock_validate_commit,
        mock_wandb,
        mock_runpod,
    ):
        """Test start_cloud_training with valid commit hash."""
        # Setup mocks
        mock_validate_commit.return_value = "4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45"
        mock_resolve_gpu.return_value = "gpu_id_123"
        mock_extract_name.return_value = "test_config"
        mock_wandb_init.return_value = ("http://test.url", "run_id_123")
        mock_runpod.create_pod.return_value = {"id": "pod_123"}

        # Mock wandb run object
        mock_run = MagicMock()
        mock_run.name = "test_name"
        mock_wandb.run = mock_run

        # Call function
        result = start_cloud_training(
            "config.py",
            commit_hash="4cc6ddb",
        )

        # Verify commit validation was called
        mock_validate_commit.assert_called_once_with("4cc6ddb")

        # Verify pod creation was called with correct docker script
        mock_runpod.create_pod.assert_called_once()
        call_args = mock_runpod.create_pod.call_args[1]
        docker_args = call_args["docker_args"]

        # The docker script should contain the git checkout command
        self.assertIn(
            "git checkout 4cc6ddbb679d1a00efd8d2d088de68eeebd9bb45", docker_args
        )

        self.assertEqual(result, "pod_123")

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    @patch("runpod_service._validate_commit_hash")
    def test_start_cloud_training_with_invalid_commit(self, mock_validate_commit):
        """Test start_cloud_training with invalid commit hash raises error."""
        mock_validate_commit.side_effect = ValueError(
            "Invalid commit hash format: invalid"
        )

        with self.assertRaises(ValueError) as cm:
            start_cloud_training("config.py", commit_hash="invalid")

        self.assertIn("Invalid commit hash format", str(cm.exception))

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"})
    @patch("runpod_service.runpod")
    @patch("runpod_service.wandb")
    @patch("runpod_service._extract_config_name")
    @patch("runpod_service._resolve_gpu_id")
    @patch("runpod_service.init_local_wandb_and_open_browser")
    def test_start_cloud_training_without_commit(
        self,
        mock_wandb_init,
        mock_resolve_gpu,
        mock_extract_name,
        mock_wandb,
        mock_runpod,
    ):
        """Test start_cloud_training without commit hash (default behavior)."""
        # Setup mocks
        mock_resolve_gpu.return_value = "gpu_id_123"
        mock_extract_name.return_value = "test_config"
        mock_wandb_init.return_value = ("http://test.url", "run_id_123")
        mock_runpod.create_pod.return_value = {"id": "pod_123"}

        # Mock wandb run object
        mock_run = MagicMock()
        mock_run.name = "test_name"
        mock_wandb.run = mock_run

        # Call function without commit_hash
        result = start_cloud_training("config.py")

        # Verify pod creation was called
        mock_runpod.create_pod.assert_called_once()
        call_args = mock_runpod.create_pod.call_args[1]
        docker_args = call_args["docker_args"]

        # The docker script should NOT contain git checkout command
        self.assertNotIn("git checkout", docker_args)

        self.assertEqual(result, "pod_123")


class TestNoteValidation(unittest.TestCase):
    """Test note validation functionality (existing test to ensure we didn't break it)."""

    def test_validate_note_valid(self):
        """Test validation with valid note."""
        _validate_note("test-note_123")  # Should not raise

    def test_validate_note_invalid_characters(self):
        """Test validation with invalid characters raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _validate_note("test note with spaces")
        self.assertIn("invalid characters", str(cm.exception))

    def test_validate_note_empty(self):
        """Test validation with empty note."""
        _validate_note("")  # Should not raise

    def test_validate_note_none(self):
        """Test validation with None note."""
        _validate_note(None)  # Should not raise


if __name__ == "__main__":
    unittest.main()
