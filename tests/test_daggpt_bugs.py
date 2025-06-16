import torch
import pytest

from dag_model import DAGGPT, DAGGPTConfig, DAGController


def test_controller_selects_distinct_inputs():
    torch.manual_seed(0)
    controller = DAGController(hidden_dim=4)
    nodes = torch.randn(1, 3, 4)
    ctx = torch.zeros(1, 4)
    input1, input2, _ = controller(nodes, ctx, ctx)
    assert not torch.allclose(input1, input2)


def test_weight_tying():
    cfg = DAGGPTConfig(vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dag_depth=1)
    model = DAGGPT(cfg)
    w1 = model.transformer.wte.weight
    w2 = model.lm_head.weight
    assert w1.data_ptr() == w2.data_ptr()


def test_forward_block_size_assertion():
    cfg = DAGGPTConfig(vocab_size=10, block_size=2, n_layer=1, n_head=1, n_embd=8, dag_depth=1)
    model = DAGGPT(cfg)
    x = torch.randint(0, 10, (1, 3))
    with pytest.raises(AssertionError):
        model(x)
