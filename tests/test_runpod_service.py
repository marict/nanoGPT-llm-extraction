import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_service as rp


def test_start_cloud_training(monkeypatch):
    created = {}

    def fake_create_pod(**kwargs):
        created['params'] = kwargs
        return {'id': 'pod123'}

    def fake_ip_port(pod_id):
        assert pod_id == 'pod123'
        return '1.2.3.4', 22

    commands = {}

    class DummySSH:
        def __init__(self, pod_id):
            assert pod_id == 'pod123'

        def run_commands(self, cmds):
            commands['value'] = cmds

        def close(self):
            pass

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(rp.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(rp, "get_pod_ssh_ip_port", fake_ip_port)
    monkeypatch.setattr(rp, "SSHConnection", DummySSH)

    pod_id = rp.start_cloud_training("config.py")
    assert pod_id == 'pod123'
    assert commands['value'] == [
        "cd /workspace && python train.py /workspace/config.py"
    ]


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

    result = rp.visualize_dag_attention(model, SimpleTokenizer(), prompt, save_path=str(out_path))
    assert Path(result).exists()
    assert torch.allclose(model.dag.controller.last_attn.squeeze(), torch.tensor([1.0, 0.0]))
    assert torch.argmax(model.dag.controller.last_op_weights).item() == 2
