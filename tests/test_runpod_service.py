import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

# Suppress wandb deprecation warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Scope.user.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path

import runpod_service as rp


# Fixture to prevent Chrome from opening during tests
@pytest.fixture(autouse=True)
def mock_subprocess_run():
    """Mock subprocess.run globally to prevent Chrome from opening during tests."""
    with patch("subprocess.run") as mock_run:

        def mock_implementation(*args, **kwargs):
            # Just return a successful result without actually running the command
            class MockResult:
                returncode = 0
                stdout = ""
                stderr = ""

            return MockResult()

        mock_run.side_effect = mock_implementation
        yield mock_run


# --------------------------------------------------------------------- #
# Test stop_runpod
# --------------------------------------------------------------------- #
def test_stop_runpod_sdk(monkeypatch):
    """stop_runpod succeeds via the RunPod SDK."""

    # fake SDK object inside runpod_service
    class _FakeRunpod:
        api_key = None

        def stop_pod(self, pid):
            # record call for assertion
            self.called_with = pid
            return {"status": "STOPPING"}

    fake_sdk = _FakeRunpod()
    monkeypatch.setattr(rp, "runpod", fake_sdk, raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_xyz")
    monkeypatch.setenv("RUNPOD_API_KEY", "key123")

    assert rp.stop_runpod() is True
    assert fake_sdk.called_with == "pod_xyz"


def test_stop_runpod_rest(monkeypatch):
    """stop_runpod falls back to REST when SDK route missing."""
    # ensure the fake `runpod` has no stop_pod attr → triggers REST branch
    monkeypatch.setattr(rp, "runpod", object(), raising=False)

    # mock requests.post
    calls = {}

    def fake_post(*args, **kwargs):
        calls["url"] = args[0] if args else kwargs.get("url")

        class _Resp:
            status_code = 200

            def raise_for_status(self): ...

        return _Resp()

    # Create a mock requests module
    class MockRequests:
        def post(self, *args, **kwargs):
            return fake_post(*args, **kwargs)

    monkeypatch.setattr(rp, "requests", MockRequests())
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_rest")
    monkeypatch.setenv("RUNPOD_API_KEY", "key_rest")

    assert rp.stop_runpod() is True
    assert calls["url"].endswith("/pod_rest/stop")


# ---------------------------------------------------------------------------
# test start_cloud_training
# ---------------------------------------------------------------------------
def test_start_cloud_training(monkeypatch):
    created = {}

    def fake_create_pod(**kwargs):
        created["params"] = kwargs
        return {"id": "pod123"}

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    pod_id = rp.start_cloud_training("config.py")
    assert pod_id == "pod123"
    assert created["params"]["gpu_type_id"] == "gpu123"
    assert created["params"]["start_ssh"] is False

    docker_args = created["params"]["docker_args"]
    assert docker_args.startswith("bash -c")
    # Check that the config file is included in the docker args
    assert "config.py" in docker_args


# --------------------------------------------------------------------- #
# Test init_local_wandb_and_open_browser
# --------------------------------------------------------------------- #
def test_init_local_wandb_comprehensive(monkeypatch):
    """Comprehensive test of wandb initialization with various scenarios."""

    # Test 1: Successful wandb initialization
    class MockRun:
        url = "https://wandb.ai/test/project/runs/test_run_123"
        id = "test_run_123"

    class MockWandb:
        def init(self, project, name, tags, notes):
            return MockRun()

        class Settings:
            def __init__(self):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")
    assert result == (
        "https://wandb.ai/test/project/runs/test_run_123/logs",
        "test_run_123",
    )

    # Test 2: Missing API key
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")
    assert result is None

    # Test 3: Wandb initialization fails
    def mock_wandb_init(*args, **kwargs):
        raise Exception("Wandb initialization failed")

    class MockWandbFail:
        init = mock_wandb_init

        class Settings:
            def __init__(self):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandbFail())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")
    assert result is None


def test_start_cloud_training_comprehensive(monkeypatch, tmp_path):
    """Comprehensive test of cloud training with various configurations."""

    # Setup common mocks
    created_pods = []

    def fake_create_pod(**kwargs):
        created_pods.append(kwargs)
        return {"id": f"pod{len(created_pods)}"}

    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    wandb_calls = []

    def mock_init_wandb(project_name, run_id):
        wandb_calls.append({"project": project_name, "run_id": run_id})
        return ("https://wandb.ai/test/project/runs/abc123", "abc123")

    monkeypatch.setattr(rp, "init_local_wandb_and_open_browser", mock_init_wandb)

    # Test 1: Default configuration
    pod_id = rp.start_cloud_training("config/test.py")
    assert pod_id == "pod1"
    assert len(wandb_calls) == 1
    assert wandb_calls[0]["project"] == "daggpt-train"
    assert wandb_calls[0]["run_id"] == "daggpt-train"

    # Test 2: With keep-alive flag
    created_pods.clear()
    wandb_calls.clear()

    pod_id = rp.start_cloud_training("config/test.py", keep_alive=True)
    assert pod_id == "pod1"
    assert len(created_pods) == 1
    assert "--keep-alive" in created_pods[0]["docker_args"]

    # Test 3: Without keep-alive flag (explicit False)
    created_pods.clear()
    wandb_calls.clear()

    pod_id = rp.start_cloud_training("config/test.py", keep_alive=False)
    assert pod_id == "pod1"
    assert len(created_pods) == 1
    assert "--keep-alive" not in created_pods[0]["docker_args"]

    # Test 4: Custom project name from config
    config_file = tmp_path / "test_config.py"
    config_file.write_text('name = "custom-project-name"\n')

    wandb_calls.clear()
    pod_id = rp.start_cloud_training(str(config_file))
    assert len(wandb_calls) == 1
    assert wandb_calls[0]["project"] == "custom-project-name"
    assert wandb_calls[0]["run_id"] == "custom-project-name"


def test_wandb_logs_url_modification(monkeypatch):
    """Test that wandb URLs are modified to point to logs page."""

    class MockRun:
        url = "https://wandb.ai/test/project/runs/test_run_123"
        id = "test_run_123"

    class MockWandb:
        def init(self, project, name, tags, notes):
            return MockRun()

        class Settings:
            def __init__(self):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")

    assert result is not None
    wandb_url, run_id = result

    # Verify the URL includes /logs suffix
    assert wandb_url == "https://wandb.ai/test/project/runs/test_run_123/logs"
    assert run_id == "test_run_123"

    print("✅ Wandb logs URL modification working correctly")
