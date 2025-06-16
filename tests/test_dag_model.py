import torch
import torch.nn as nn
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
import pytest


@pytest.fixture(scope="module")
def small_dag_gpt():
    config = DAGGPTConfig(vocab_size=20, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=2)
    return DAGGPT(config), config


def test_dag_gpt_forward(small_dag_gpt):
    model, config = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    binary = torch.zeros(2, 4, 33)
    logits, loss, dag_out = model(x, binary=binary)
    assert logits.shape == (2, 4, 20)
    assert dag_out.shape == (2, config.n_embd)


def test_dag_node_growth_regression(monkeypatch):
    class DummyController(DAGController):
        def forward(self, nodes):
            input1 = nodes[:, -1, :]
            input2 = input1
            op_weights = torch.tensor([[1.0, 0.0]])
            return input1, input2, op_weights

    monkeypatch.setattr(dag_model, "op_funcs", dag_model.op_funcs[:2])
    dag = DifferentiableDAG(hidden_dim=4, num_ops=2, num_steps=2)
    dag.controller = DummyController(4, 2)
    out = dag([torch.ones(1, 4)])
    assert out.shape == (1, 3, 4)
    assert torch.allclose(out[:, -1, :], torch.full((1, 4), 4.0))


def test_op_functions():
    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([1.0, 2.0])
    assert torch.allclose(multiply(x, y), x * y)
    assert torch.allclose(subtract(x, y), x - y)
    assert torch.allclose(divide(x, y), x / y)


def test_dag_backward_flow(small_dag_gpt):
    model, _ = small_dag_gpt
    x = torch.randint(0, 20, (2, 4))
    binary = torch.zeros(2, 4, 33)
    _, _, dag_out = model(x, binary=binary)
    loss = dag_out.sum()
    loss.backward()
    grad = model.dag.controller.op_selector.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_predict_number(monkeypatch):
    from numeric_tokenizer import NumericTokenizer

    tok = NumericTokenizer()
    tokens, binary = tok.encode("7")

    cfg = DAGGPTConfig(
        vocab_size=tok.next_id,
        block_size=len(tokens),
        n_layer=1,
        n_head=1,
        n_embd=8,
        dag_depth=1,
    )
    model = DAGGPT(cfg)

    class DummyHead(nn.Module):
        def __init__(self, token_id, vocab_size):
            super().__init__()
            self.token_id = token_id
            self.vocab_size = vocab_size

        def forward(self, x):
            b = x.size(0)
            logits = torch.zeros(b, self.vocab_size)
            logits[:, self.token_id] = 5.0
            return logits

    model.lm_head = DummyHead(tokens[0], cfg.vocab_size)

    number = model.predict_number(
        torch.tensor(tokens).unsqueeze(0),
        binary=torch.tensor(binary).unsqueeze(0),
        tokenizer=tok,
    )

    assert number == 7.0

