import sys
from pathlib import Path

import pytest
pytest.importorskip("runpod")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_service as rp


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
    assert docker_args.startswith(
        f"bash -c '[ -d repo ] && git -C repo pull || git clone {rp.REPO_URL} repo"
    )
    assert f"python train.py config.py --wandb_project={rp.POD_NAME}'" in docker_args


def test_visualize_dag_attention(tmp_path):
    import torch

    import dag_model
    from dag_model import DAGGPT, DAGController, DAGGPTConfig

    class DummyController(DAGController):
        def forward(self, nodes, operand_ctx, op_ctx):
            self.last_attn = torch.tensor([[1.0, 0.0]])
            input1 = nodes[:, 0, :]
            input2 = nodes[:, 1, :]
            weights = torch.zeros(1, len(dag_model.op_funcs))
            weights[0, 2] = 1.0
            self.last_op_weights = weights
            return input1, input2, self.last_op_weights

    prompt = "2 3"
    tokens = [2, 3]
    cfg = DAGGPTConfig(
        vocab_size=10,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=4,
        dag_depth=1,
    )
    model = DAGGPT(cfg)
    model.dag.controller = DummyController(cfg.n_embd)
    out_path = tmp_path / "viz.png"

    class SimpleTokenizer:
        def encode(self, text):
            return tokens

    result = rp.visualize_dag_attention(
        model, SimpleTokenizer(), prompt, save_path=str(out_path)
    )
    assert Path(result).exists()
    assert torch.allclose(
        model.dag.controller.last_attn.squeeze(), torch.tensor([1.0, 0.0])
    )
    assert torch.argmax(model.dag.controller.last_op_weights).item() == 2
