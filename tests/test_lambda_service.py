import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lambda_service as ls


class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


def test_start_cloud_training(monkeypatch):
    created = {}

    def fake_post(url, json, headers):
        created['payload'] = json
        return DummyResponse({"data": {"instance_ids": ["inst123"]}})

    statuses = [
        DummyResponse({"data": {"status": "booting"}}),
        DummyResponse({"data": {"status": "terminated"}}),
    ]

    def fake_get(url, headers):
        return statuses.pop(0)

    monkeypatch.setenv("LAMBDA_API_KEY", "key")
    monkeypatch.setenv("LAMBDA_SSH_KEY", "ssh")
    monkeypatch.setattr(ls.requests, "post", fake_post)
    monkeypatch.setattr(ls.requests, "get", fake_get)
    monkeypatch.setattr(ls.time, "sleep", lambda x: None)
    inst_id = ls.start_cloud_training("config.py")
    assert inst_id == "inst123"
    assert "/workspace/config.py" in created["payload"].get("user_data", "")


def test_start_cloud_training_default_config(monkeypatch):
    created = {}

    def fake_post(url, json, headers):
        created['payload'] = json
        return DummyResponse({"data": {"instance_ids": ["inst123"]}})

    statuses = [
        DummyResponse({"data": {"status": "booting"}}),
        DummyResponse({"data": {"status": "terminated"}}),
    ]

    def fake_get(url, headers):
        return statuses.pop(0)

    monkeypatch.setenv("LAMBDA_API_KEY", "key")
    monkeypatch.setenv("LAMBDA_SSH_KEY", "ssh")
    monkeypatch.setattr(ls.requests, "post", fake_post)
    monkeypatch.setattr(ls.requests, "get", fake_get)
    monkeypatch.setattr(ls.time, "sleep", lambda x: None)
    inst_id = ls.start_cloud_training("config/train_chatgpt2.py")
    assert inst_id == "inst123"
    assert "/workspace/config/train_chatgpt2.py" in created["payload"].get("user_data", "")


def test_start_cloud_training_accepts_201(monkeypatch):
    created = {}

    def fake_post(url, json, headers):
        created['payload'] = json
        return DummyResponse({"data": {"instance_ids": ["inst456"]}}, status=201)

    statuses = [
        DummyResponse({"data": {"status": "booting"}}),
        DummyResponse({"data": {"status": "terminated"}}),
    ]

    def fake_get(url, headers):
        return statuses.pop(0)

    monkeypatch.setenv("LAMBDA_API_KEY", "key")
    monkeypatch.setenv("LAMBDA_SSH_KEY", "ssh")
    monkeypatch.setattr(ls.requests, "post", fake_post)
    monkeypatch.setattr(ls.requests, "get", fake_get)
    monkeypatch.setattr(ls.time, "sleep", lambda x: None)
    inst_id = ls.start_cloud_training("config.py")
    assert inst_id == "inst456"
    assert "/workspace/config.py" in created["payload"].get("user_data", "")


def test_visualize_dag_attention(tmp_path):
    from dag_model import DAGGPT, DAGGPTConfig, DAGController
    import dag_model
    import torch

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

    result = ls.visualize_dag_attention(model, SimpleTokenizer(), prompt, save_path=str(out_path))
    assert Path(result).exists()
    assert torch.allclose(model.dag.controller.last_attn.squeeze(), torch.tensor([1.0, 0.0]))
    assert torch.argmax(model.dag.controller.last_op_weights).item() == 2

