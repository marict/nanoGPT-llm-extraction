import types
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import runpod_service as rp


class DummyEndpoint:
    def __init__(self, endpoint_id):
        self.endpoint_id = endpoint_id
        self.called = False

    def run_sync(self, data):
        self.called = True
        return {"output": "ok"}


def test_run_inference(monkeypatch):
    dummy = DummyEndpoint("123")
    monkeypatch.setattr(
        rp, "runpod", types.SimpleNamespace(Endpoint=lambda x: dummy, api_key=None)
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    result = rp.run_inference("hi", endpoint_id="123")
    assert result == "ok"
    assert dummy.called


def test_start_cloud_training(monkeypatch):
    created = {}

    def fake_create_pod(**kwargs):
        created.update(kwargs)
        return {"id": "pod123"}

    statuses = [{"state": "STARTING"}, {"state": "STOPPED"}]

    def fake_get_pod(pod_id):
        return statuses.pop(0)

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(
        rp,
        "runpod",
        types.SimpleNamespace(
            create_pod=fake_create_pod,
            get_pod=fake_get_pod,
            Endpoint=None,
            api_key=None,
        ),
    )
    monkeypatch.setattr(rp.time, "sleep", lambda x: None)
    pod_id = rp.start_cloud_training("config.py")
    assert pod_id == "pod123"
    assert "docker_args" in created and "config.py" in created["docker_args"]


def test_start_cloud_training_default_config(monkeypatch):
    created = {}

    def fake_create_pod(**kwargs):
        created.update(kwargs)
        return {"id": "pod123"}

    statuses = [{"state": "STARTING"}, {"state": "STOPPED"}]

    def fake_get_pod(pod_id):
        return statuses.pop(0)

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(
        rp,
        "runpod",
        types.SimpleNamespace(
            create_pod=fake_create_pod,
            get_pod=fake_get_pod,
            Endpoint=None,
            api_key=None,
        ),
    )
    monkeypatch.setattr(rp.time, "sleep", lambda x: None)
    pod_id = rp.start_cloud_training("config/train_chatgpt2.py")
    assert pod_id == "pod123"
    assert "config/train_chatgpt2.py" in created.get("docker_args", "")


def test_visualize_dag_attention(tmp_path):
    from dag_model import DAGGPT, DAGGPTConfig, DAGController
    from numeric_tokenizer import NumericTokenizer
    import torch

    class DummyController(DAGController):
        def forward(self, nodes):
            self.last_attn = torch.tensor([[1.0, 0.0]])
            input1 = nodes[:, 0, :]
            input2 = nodes[:, 1, :]
            self.last_op_weights = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
            return input1, input2, self.last_op_weights

    tok = NumericTokenizer()
    prompt = "2 3"
    tokens, binary = tok.encode(prompt)
    cfg = DAGGPTConfig(
        vocab_size=tok.next_id,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=4,
        dag_depth=1,
    )
    model = DAGGPT(cfg)
    model.dag.controller = DummyController(cfg.n_embd, cfg.dag_num_ops)
    out_path = tmp_path / "viz.png"
    result = rp.visualize_dag_attention(model, tok, prompt, save_path=str(out_path))
    assert os.path.exists(result)
    assert torch.allclose(model.dag.controller.last_attn.squeeze(), torch.tensor([1.0, 0.0]))
    assert torch.argmax(model.dag.controller.last_op_weights).item() == 2

