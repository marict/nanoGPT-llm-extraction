"""Test git commit logging functionality."""

import subprocess
from io import StringIO
from unittest.mock import MagicMock, patch

from training_utils import log_git_commit_info


def test_log_git_commit_info_long_commit_message():
    """Test git logging with a long commit message that gets truncated."""
    long_message = "This is a very long commit message that should be truncated at 120 characters to prevent issues with very long commit messages that exceed the display limit set for readability and to avoid overwhelming the logs with extremely verbose commit descriptions"

    mock_outputs = {
        ("git", "rev-parse", "HEAD"): "a1b2c3d4e5f6789012345678901234567890abcd\n",
        ("git", "branch", "--show-current"): "feature-branch\n",
        ("git", "log", "-1", "--format=%s"): long_message + "\n",
    }

    def mock_subprocess_run(cmd, *args, **kwargs):
        key = tuple(cmd)
        if key in mock_outputs:
            mock_result = MagicMock()
            mock_result.stdout = mock_outputs[key]
            return mock_result
        raise subprocess.CalledProcessError(1, cmd)

    with patch("training_utils.subprocess.run", side_effect=mock_subprocess_run):
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            log_git_commit_info()

        output = captured_output.getvalue()
        # Should contain truncated message with "..."
        expected_msg = long_message[:120] + "..."
        assert expected_msg in output


def test_log_git_commit_info_branch_failure():
    """Test git logging when branch command fails."""
    mock_outputs = {
        ("git", "rev-parse", "HEAD"): "a1b2c3d4e5f6789012345678901234567890abcd\n",
        ("git", "log", "-1", "--format=%s"): "Test commit\n",
    }

    def mock_subprocess_run(cmd, *args, **kwargs):
        key = tuple(cmd)
        if key in mock_outputs:
            mock_result = MagicMock()
            mock_result.stdout = mock_outputs[key]
            return mock_result
        # Branch command fails
        if "branch" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        raise subprocess.CalledProcessError(1, cmd)

    with patch("training_utils.subprocess.run", side_effect=mock_subprocess_run):
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            log_git_commit_info()

        output = captured_output.getvalue()


def test_log_git_commit_info_git_not_available():
    """Test git logging when git is not available."""

    def mock_subprocess_run(cmd, *args, **kwargs):
        raise FileNotFoundError("git not found")

    with patch("training_utils.subprocess.run", side_effect=mock_subprocess_run):
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            log_git_commit_info()

        output = captured_output.getvalue()
        assert "ERROR: Git command not found" in output


def test_log_git_commit_info_timeout():
    """Test git logging when git commands timeout."""

    def mock_subprocess_run(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)

    with patch("training_utils.subprocess.run", side_effect=mock_subprocess_run):
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            log_git_commit_info()

        output = captured_output.getvalue()
        assert "ERROR: Git command timed out" in output


def test_log_git_commit_info_all_commands_fail():
    """Test git logging when all git commands fail."""

    def mock_subprocess_run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    with patch("training_utils.subprocess.run", side_effect=mock_subprocess_run):
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            log_git_commit_info()

        output = captured_output.getvalue()
        assert "ERROR: Git command failed" in output


def test_log_git_commit_info_in_actual_repo():
    """Test git logging in the actual repository (integration test)."""
    # This test runs against the actual git repo
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        log_git_commit_info()

    output = captured_output.getvalue()
    # Should contain the basic format even if we can't predict exact values
    assert "Git repository analysis:" in output
    assert "Commit hash:" in output
    assert "Branch:" in output
    assert "Message:" in output
