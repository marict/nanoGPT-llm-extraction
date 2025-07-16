import torch

from data.dagset.streaming import OP_NAMES, DAGStructureDataset


def test_post_identity_rows_are_identity():
    depth = 4
    # Force identity_cutoff_p=1.0 to guarantee early identity
    dataset = DAGStructureDataset(max_depth=depth, seed=0, identity_cutoff_p=1.0)
    text, structure = dataset.generate_structure_example(depth, seed=0)

    op_probs = structure["operation_probs"]  # shape (depth, n_ops)
    picked = op_probs.argmax(dim=-1)
    identity_idx = OP_NAMES.index("identity")

    # After the first identity row, every following row should also be identity
    first_identity = (picked == identity_idx).nonzero(as_tuple=True)[0][0].item()

    # All rows at/after the first identity should be identity (including the
    # trivial case where identity is already the last row).
    assert torch.all(
        picked[first_identity:] == identity_idx
    ), "Rows after the initial identity must be identity-padded"
