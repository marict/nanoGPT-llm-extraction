import pytest
import torch

from data.dagset.streaming import DAGStructureDataset, generate_random_dag_plan
from models.dag_model import OP_NAMES
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel

SUBSET_OPS = ["add", "subtract", "identity"]


def _make_dummy_predictor(depth: int = 2, n_embd: int = 32, n_head: int = 2):
    """Utility: build a tiny stand-alone predictor that only predicts SUBSET_OPS."""

    cfg = PredictorOnlyConfig(
        vocab_size=50,
        n_embd=n_embd,
        n_head=n_head,
        dropout=0.0,
        bias=False,
        dag_depth=depth,
        sequence_length=16,
        softmax_temperature=5.0,
        op_names=SUBSET_OPS,
    )

    return PredictorOnlyModel(cfg)


def test_predictor_only_predicts_subset_ops():
    """Model should only allocate logits for the configured subset of operations."""

    model = _make_dummy_predictor()
    model.eval()

    batch, seq_len = 2, model.config.sequence_length
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        _, _, op_probs = model(input_ids)  # (B, T, depth, len(SUBSET_OPS))

    # Predictor should emit full OP_NAMES dimension but with zero probability mass
    assert op_probs.shape[-1] == len(OP_NAMES)

    disallowed_indices = {
        idx for idx in range(len(OP_NAMES)) if OP_NAMES[idx] not in SUBSET_OPS
    }
    probs_sum_disallowed = op_probs[..., list(disallowed_indices)].sum()
    assert (
        probs_sum_disallowed.item() < 1e-5
    ), "Predictor assigned probability mass to operations outside the configured subset."


def test_generate_random_dag_plan_subset():
    """Random DAG plan generator must only sample from the allowed subset."""

    depth = 5
    init_vals, ops = generate_random_dag_plan(
        depth=depth,
        num_initial_values=depth + 1,
        seed=123,
        allowed_operations=SUBSET_OPS,
    )

    assert 1 <= len(ops) <= depth
    assert all(op in SUBSET_OPS for op in ops), "Found operation outside subset."


def test_dagstructure_dataset_subset_ops():
    """The streaming dataset should restrict operations to the provided subset."""

    depth = 3
    dataset = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,  # deterministic for test
        allowed_operations=SUBSET_OPS,
    )

    # Generate a single example
    _text, structure = dataset.generate_structure_example(depth, seed=0)

    # operation_probs shape: (depth, len(OP_NAMES)) – identify non-zero indices
    op_one_hot = structure["operation_probs"]
    picked_indices = op_one_hot.argmax(dim=-1)

    allowed_indices = [OP_NAMES.index(op) for op in SUBSET_OPS]

    assert set(picked_indices.tolist()).issubset(
        set(allowed_indices)
    ), "Dataset produced operations outside the allowed subset."
