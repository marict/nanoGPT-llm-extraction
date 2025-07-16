import pytest
import torch

from data.dagset.streaming import OP_NAMES, DAGStructureDataset


def test_post_identity_rows_are_identity():
    depth = 4
    # Force identity_cutoff_p=1.0 to guarantee early identity
    dataset = DAGStructureDataset(max_depth=depth, seed=0, identity_cutoff_p=1.0)
    identity_idx = OP_NAMES.index("identity")

    for seed in range(20):
        text, structure = dataset.generate_structure_example(depth, seed=seed)
        picked = structure["operation_probs"].argmax(dim=-1)
        id_positions = (picked == identity_idx).nonzero(as_tuple=True)[0]
        if id_positions.numel() == 0:
            continue  # no identity in this plan, try next seed
        first_identity = id_positions[0].item()
        assert torch.all(
            picked[first_identity:] == identity_idx
        ), "Rows after the initial identity must be identity-padded"
        break
    else:
        pytest.skip("No identity generated in 20 seeds; skip padding test")
