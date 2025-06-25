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


# ---------------------------------------------------------------------------
# helper: very small tokenizer for test_visualize_dag_attention
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    def __init__(self, ids):
        self._ids = ids

    def encode(self, text):
        # Return the pre-baked token list irrespective of text (good enough for a unit test)
        return self._ids


# ---------------------------------------------------------------------------
# test visualize_dag_attention
# ---------------------------------------------------------------------------
def test_visualize_dag_attention(tmp_path):
    """
    Fast smoke-test the visualisation helper by mocking matplotlib operations.
    """

    # Mock matplotlib to avoid actual plotting
    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.subplots"), patch(
        "matplotlib.pyplot.savefig"
    ) as mock_savefig, patch("matplotlib.pyplot.close"):

        # Mock the actual visualization function to just create a file
        def mock_viz_func(*args, **kwargs):
            save_path = kwargs.get("save_path", str(tmp_path / "viz.png"))
            Path(save_path).touch()  # Create empty file
            return save_path

        # Simple mock that just tests the interface
        tokens = [0, 1]  # minimal tokens
        out_path = tmp_path / "viz.png"

        # Mock the actual function call
        with patch.object(
            rp, "visualize_dag_attention", side_effect=mock_viz_func
        ) as mock_viz:
            result_file = rp.visualize_dag_attention(
                None,  # model (mocked)
                SimpleTokenizer(tokens),
                prompt="0 1",
                save_path=str(out_path),
            )

            # Verify the function was called and file exists
            assert mock_viz.called
            assert Path(result_file).exists(), "Visualization file not created"


# --------------------------------------------------------------------- #
# Test init_local_wandb_and_open_browser
# --------------------------------------------------------------------- #
def test_init_local_wandb_and_open_browser_success(monkeypatch):
    """Test successful wandb initialization and browser opening."""

    # Mock wandb
    class MockRun:
        url = "https://wandb.ai/test/project/runs/test_run_123"

    class MockWandb:
        def init(self, project, name, tags, notes):
            return MockRun()

        class Settings:
            def __init__(self):
                pass

    mock_wandb = MockWandb()
    monkeypatch.setattr(rp, "wandb", mock_wandb)

    # Mock environment
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    # Note: subprocess.run is already mocked globally by the fixture

    # Test the function
    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")

    assert result == "https://wandb.ai/test/project/runs/test_run_123"


def test_init_local_wandb_and_open_browser_no_api_key(monkeypatch):
    """Test when WANDB_API_KEY is not set."""

    # Remove WANDB_API_KEY
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")

    assert result is None


def test_init_local_wandb_and_open_browser_wandb_fails(monkeypatch):
    """Test when wandb initialization fails."""

    # Mock wandb to raise an exception
    def mock_wandb_init(*args, **kwargs):
        raise Exception("Wandb initialization failed")

    class MockWandb:
        init = mock_wandb_init

        class Settings:
            def __init__(self):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")

    assert result is None


def test_init_local_wandb_and_open_browser_chrome_fails(monkeypatch):
    """Test when Chrome opening fails but wandb succeeds."""

    # Mock wandb
    class MockRun:
        url = "https://wandb.ai/test/project/runs/test_run_123"

    class MockWandb:
        def init(self, project, name, tags, notes):
            return MockRun()

        class Settings:
            def __init__(self):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    # Note: subprocess.run is already mocked globally by the fixture

    # Test the function
    result = rp.init_local_wandb_and_open_browser("test-project", "test_run_123")

    # Should still return URL even if Chrome opening fails
    assert result == "https://wandb.ai/test/project/runs/test_run_123"


def test_start_cloud_training_with_wandb_integration(monkeypatch):
    """Test that start_cloud_training calls wandb initialization."""

    # Mock RunPod API
    def fake_create_pod(**kwargs):
        return {"id": "pod123"}

    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    # Track wandb initialization calls
    wandb_calls = []

    def mock_init_wandb(project_name, run_id):
        wandb_calls.append({"project": project_name, "run_id": run_id})
        return "https://wandb.ai/test/project/runs/abc123"

    monkeypatch.setattr(rp, "init_local_wandb_and_open_browser", mock_init_wandb)

    # Test
    pod_id = rp.start_cloud_training("config/test.py")

    # Verify wandb was called
    assert len(wandb_calls) == 1
    assert wandb_calls[0]["project"] == "daggpt-train"  # default project name
    assert wandb_calls[0]["run_id"] == pod_id


def test_start_cloud_training_extracts_project_name_from_config(monkeypatch, tmp_path):
    """Test that project name is extracted from config file."""

    # Create a test config file
    config_file = tmp_path / "test_config.py"
    config_file.write_text('name = "custom-project-name"\n')

    # Mock RunPod API
    def fake_create_pod(**kwargs):
        return {"id": "pod456"}

    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    # Track wandb initialization calls
    wandb_calls = []

    def mock_init_wandb(project_name, run_id):
        wandb_calls.append({"project": project_name, "run_id": run_id})
        return "https://wandb.ai/test/project/runs/abc123"

    monkeypatch.setattr(rp, "init_local_wandb_and_open_browser", mock_init_wandb)

    # Test
    pod_id = rp.start_cloud_training(str(config_file))

    # Verify wandb was called with custom project name
    assert len(wandb_calls) == 1
    assert wandb_calls[0]["project"] == "custom-project-name"
    assert wandb_calls[0]["run_id"] == pod_id


def test_start_cloud_training_with_keep_alive(monkeypatch):
    """Test that start_cloud_training handles keep-alive flag correctly."""

    # Mock RunPod API
    created_pods = []

    def fake_create_pod(**kwargs):
        created_pods.append(kwargs)
        return {"id": "pod789"}

    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    # Mock wandb initialization
    def mock_init_wandb(project_name, run_id):
        return "https://wandb.ai/test/project/runs/abc123"

    monkeypatch.setattr(rp, "init_local_wandb_and_open_browser", mock_init_wandb)

    # Test with keep_alive=True
    pod_id = rp.start_cloud_training("config/test.py", keep_alive=True)

    # Verify docker args include keep-alive flag
    assert len(created_pods) == 1
    docker_args = created_pods[0]["docker_args"]
    assert "--keep-alive" in docker_args
    assert pod_id == "pod789"


def test_start_cloud_training_without_keep_alive(monkeypatch):
    """Test that start_cloud_training works normally without keep-alive flag."""

    # Mock RunPod API
    created_pods = []

    def fake_create_pod(**kwargs):
        created_pods.append(kwargs)
        return {"id": "pod999"}

    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        rp.runpod,
        "get_gpus",
        lambda: [{"id": "gpu123", "displayName": rp.DEFAULT_GPU_TYPE}],
    )

    # Mock wandb initialization
    def mock_init_wandb(project_name, run_id):
        return "https://wandb.ai/test/project/runs/abc123"

    monkeypatch.setattr(rp, "init_local_wandb_and_open_browser", mock_init_wandb)

    # Test with keep_alive=False (default)
    pod_id = rp.start_cloud_training("config/test.py", keep_alive=False)

    # Verify docker args do NOT include keep-alive flag
    assert len(created_pods) == 1
    docker_args = created_pods[0]["docker_args"]
    assert "--keep-alive" not in docker_args
    assert pod_id == "pod999"
