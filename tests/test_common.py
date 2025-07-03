"""Common test utilities and fixtures to reduce redundancy across test files."""

import pytest
import torch

from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig

# Set random seeds for reproducible tests
torch.manual_seed(42)


# Common model configurations
TINY_CONFIG = GPTConfig(
    vocab_size=20,
    block_size=4,
    n_layer=1,
    n_head=1,
    n_embd=8,
    dag_depth=2,
    dropout=0.0,
    bias=False,
)

SMALL_CONFIG = GPTConfig(
    vocab_size=50,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=32,
    dag_depth=2,
    dropout=0.0,
    bias=False,
)

STANDARD_CONFIG = GPTConfig(
    vocab_size=100,
    block_size=16,
    n_layer=2,
    n_head=4,
    n_embd=64,
    dag_depth=2,
    dropout=0.0,
    bias=False,
)


@pytest.fixture(scope="module")
def tiny_model():
    """Tiny model for quick tests."""
    return GPT(TINY_CONFIG), TINY_CONFIG


@pytest.fixture(scope="module")
def small_model():
    """Small model for basic tests."""
    return GPT(SMALL_CONFIG), SMALL_CONFIG


@pytest.fixture(scope="module")
def standard_model():
    """Standard model for comprehensive tests."""
    return GPT(STANDARD_CONFIG), STANDARD_CONFIG


@pytest.fixture
def sample_batch_tiny():
    """Sample batch for tiny model."""
    return (
        torch.randint(0, TINY_CONFIG.vocab_size, (2, TINY_CONFIG.block_size)),
        torch.randint(0, TINY_CONFIG.vocab_size, (2, TINY_CONFIG.block_size)),
    )


@pytest.fixture
def sample_batch_small():
    """Sample batch for small model."""
    return (
        torch.randint(0, SMALL_CONFIG.vocab_size, (2, SMALL_CONFIG.block_size)),
        torch.randint(0, SMALL_CONFIG.vocab_size, (2, SMALL_CONFIG.block_size)),
    )


def assert_valid_forward_pass(model, config, input_ids, target_ids=None):
    """Common assertions for forward pass validation."""
    if target_ids is not None:
        logits, loss = model(input_ids, target_ids)
        assert loss is not None
        assert torch.isfinite(loss)
        expected_seq_len = input_ids.shape[1]
    else:
        logits, loss = model(input_ids)
        assert loss is None
        # Standard GPT (dag_depth=0) returns only last token when generating
        expected_seq_len = (
            1 if getattr(config, "dag_depth", 0) == 0 else input_ids.shape[1]
        )

    batch_size = input_ids.shape[0]
    assert logits.shape == (batch_size, expected_seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def assert_valid_node_values(model):
    """Common assertions for node values validation."""
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)
    assert isinstance(node_values, list)
    assert len(node_values) > 0
    for val in node_values:
        assert isinstance(val, float)
        assert torch.isfinite(torch.tensor(val))


def assert_valid_logging(model):
    """Common assertions for logging functionality."""
    logger = DAGLogger()
    logger.compute_log_statistics(model)

    # Test console logging works
    try:
        logger.format_console_logging(model)
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")

    # Test wandb logging dict
    wandb_dict = logger.get_wandb_logging_dict(model)
    assert isinstance(wandb_dict, dict)

    # Test extra values
    extra_vals = logger.get_extra_vals(model)
    assert isinstance(extra_vals, dict)

    return logger, wandb_dict, extra_vals


def setup_gradient_tracking_test(model, config):
    """Common setup for gradient tracking tests."""
    model.train()
    logger = DAGLogger()

    batch_size = 2
    seq_len = config.block_size
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass first to create operation probabilities
    _, loss = model(input_ids, target_ids)

    # Set up gradient tracking after forward pass
    logger.setup_gradient_tracking(model)
    logger.update_gradient_tracking(model)

    # Backward pass to create gradients
    loss.backward()

    # Compute statistics to populate logger.logging_data
    logger.compute_log_statistics(model)

    return logger, loss, input_ids, target_ids


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def __init__(self, vocab_size=50):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, y=None):
        batch_size, seq_len = x.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        loss = torch.tensor(1.0) if y is not None else None
        return logits, loss

    def generate(self, x, max_new_tokens=1, **kwargs):
        """Return input tensor with dummy extension."""
        batch_size, seq_len = x.shape
        new_tokens = torch.zeros(
            (batch_size, max_new_tokens), dtype=torch.long, device=x.device
        )
        return torch.cat([x, new_tokens], dim=1)

    def eval(self):
        return self

    def train(self):
        return self

    def to(self, device):
        return self


def mock_encode(text):
    """Mock encode function for testing."""
    return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)


def mock_decode(tokens):
    """Mock decode function for testing."""
    return f"decoded_{len(tokens)}_tokens"
