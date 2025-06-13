import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import dag_model
from dag_model import (
    DAGGPT,
    DAGGPTConfig,
    DifferentiableDAG,
    DAGController,
    multiply,
    subtract,
    divide,
)


def test_dag_gpt_forward():
    config = DAGGPTConfig(vocab_size=20, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_steps=2)
    model = DAGGPT(config)
    x = torch.randint(0, 20, (1, 4))
    binary = torch.zeros(1, 4, 33)
    logits, loss, dag_out = model(x, binary=binary)
    assert logits.shape == (1, 4, 20)
    assert dag_out.shape[-1] == config.n_embd


def test_dag_node_growth_regression(monkeypatch):
    class DummyController(DAGController):
        def forward(self, nodes):
            input1 = nodes[-1]
            input2 = nodes[-1]
            op_weights = torch.tensor([1.0, 0.0])
            return input1, input2, op_weights

    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    dag = DifferentiableDAG(hidden_dim=4, num_ops=2, num_steps=2)
    dag.controller = DummyController(4, 2)
    out = dag([torch.ones(4)])
    assert out.shape == (3, 4)
    assert torch.allclose(out[-1], torch.full((4,), 4.0))


def test_op_functions():
    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([1.0, 2.0])
    assert torch.allclose(multiply(x, y), x * y)
    assert torch.allclose(subtract(x, y), x - y)
    assert torch.allclose(divide(x, y), x / y)
