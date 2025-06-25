import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path

import torch

import runpod_service as rp
from dag_model import GPT, DAGController, GPTConfig, op_funcs


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
    Smoke-test the visualisation helper.
    A dummy controller returns a fixed attention pattern and op-weights;
    we check that these values are reflected in the model after running
    `visualize_dag_attention`.
    """

    # ---------------------------------------------------------------------
    # dummy controller that produces deterministic attentions / op-weights
    # ---------------------------------------------------------------------
    class DummyController(DAGController):
        def __init__(self, hidden_dim, n_ops, temperature=1.0):
            super().__init__(hidden_dim, n_ops, temperature)

        def forward(self, embeds, operand_ctx, op_ctx):
            # two existing nodes → length-2 distributions
            att1 = torch.tensor([[1.0, 0.0]], device=embeds.device)  # (B=1 , N=2)
            att2 = torch.tensor([[0.0, 1.0]], device=embeds.device)  # (B=1 , N=2)

            # one-hot op selection: use the 3rd op (index 2)
            op_w = torch.zeros(1, len(op_funcs), device=embeds.device)
            op_w[0, 2] = 1.0

            # store for assertions
            self.last_attn = att1.detach()
            self.last_op_weights = op_w.detach()
            return att1, att2, op_w

    # ---------------------------------------------------------------------
    # tiny DAG-GPT model
    # ---------------------------------------------------------------------
    tokens = [2, 3]  # pretend these came from the prompt
    cfg = GPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=4,
        dag_depth=1,
    )
    model = GPT(cfg)
    model.dag.controller = DummyController(cfg.n_embd, n_ops=len(op_funcs))

    # ---------------------------------------------------------------------
    # run visualiser
    # ---------------------------------------------------------------------
    out_path = tmp_path / "viz.png"

    result_file = rp.visualize_dag_attention(
        model,
        SimpleTokenizer(tokens),
        prompt="2 3",
        save_path=str(out_path),
    )

    # ---------------------------------------------------------------------
    # assertions
    # ---------------------------------------------------------------------
    assert Path(result_file).exists(), "Visualization file not created"
    assert torch.allclose(
        model.dag.controller.last_attn.squeeze(),
        torch.tensor([1.0, 0.0]),
        atol=1e-5,
        rtol=0,
    ), "Last attention distribution mismatch"
    assert (
        torch.argmax(model.dag.controller.last_op_weights).item() == 2
    ), "Last op weights mismatch"


# --------------------------------------------------------------------- #
# Test init_local_wandb_and_open_browser
# --------------------------------------------------------------------- #
def test_init_local_wandb_and_open_browser_success(monkeypatch):
    """Test successful wandb initialization and browser opening."""

    # Mock wandb
    class MockRun:
        url = "https://wandb.ai/test/project/runs/test_run_123"

    class MockWandb:
        def init(self, project, name, tags, notes, settings):
            return MockRun()

        class Settings:
            def __init__(self, start_method):
                pass

    mock_wandb = MockWandb()
    monkeypatch.setattr(rp, "wandb", mock_wandb)

    # Mock environment
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    # Mock subprocess to simulate successful Chrome opening
    def mock_subprocess_run(cmd, check=False, capture_output=False, timeout=None):
        if "chrome" in cmd[0].lower() or "google" in cmd[0].lower():
            return  # Success (no exception)
        raise FileNotFoundError("Command not found")

    monkeypatch.setattr(rp.subprocess, "run", mock_subprocess_run)

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
            def __init__(self, start_method):
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
        def init(self, project, name, tags, notes, settings):
            return MockRun()

        class Settings:
            def __init__(self, start_method):
                pass

    monkeypatch.setattr(rp, "wandb", MockWandb())
    monkeypatch.setenv("WANDB_API_KEY", "test_key_123")

    # Mock subprocess to always fail
    def mock_subprocess_run(cmd, check=False, capture_output=False, timeout=None):
        raise FileNotFoundError("Chrome not found")

    monkeypatch.setattr(rp.subprocess, "run", mock_subprocess_run)

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
