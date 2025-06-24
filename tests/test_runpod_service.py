import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path

import torch

import runpod_service as rp
from dag_model import GPT, DAGController, GPTConfig, op_funcs


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
