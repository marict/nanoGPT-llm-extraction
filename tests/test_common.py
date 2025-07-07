"""Common test utilities and fixtures to reduce redundancy across test files."""

import pytest
import torch

from dag_logger import DAGLogger
from models.dag_model import GPT, GPTConfig

# Set random seeds for reproducible tests
torch.manual_seed(42)


# Optimized model configurations for faster testing
TINY_CONFIG = GPTConfig(
    vocab_size=16,  # Reduced from 20
    block_size=3,  # Reduced from 4
    n_layer=1,
    n_head=1,
    n_embd=8,
    dag_depth=1,  # Reduced from 2
    dropout=0.0,
    bias=False,
)

SMALL_CONFIG = GPTConfig(
    vocab_size=32,  # Reduced from 50
    block_size=6,  # Reduced from 8
    n_layer=1,  # Reduced from 2
    n_head=2,
    n_embd=16,  # Reduced from 32
    dag_depth=2,
    dropout=0.0,
    bias=False,
)

STANDARD_CONFIG = GPTConfig(
    vocab_size=64,  # Reduced from 100
    block_size=8,  # Reduced from 16
    n_layer=2,
    n_head=2,  # Reduced from 4
    n_embd=32,  # Reduced from 64
    dag_depth=2,
    dropout=0.0,
    bias=False,
)


@pytest.fixture(scope="function")  # Changed to session scope for maximum reuse
def tiny_model():
    """Tiny model for quick tests."""
    torch.manual_seed(42)  # Ensure deterministic model initialization
    return GPT(TINY_CONFIG), TINY_CONFIG


@pytest.fixture(scope="function")  # Changed to session scope for maximum reuse
def small_model():
    """Small model for basic tests."""
    torch.manual_seed(42)  # Ensure deterministic model initialization
    return GPT(SMALL_CONFIG), SMALL_CONFIG


@pytest.fixture(scope="function")  # Keep module scope as it's less frequently used
def standard_model():
    """Standard model for comprehensive tests."""
    torch.manual_seed(42)  # Ensure deterministic model initialization
    return GPT(STANDARD_CONFIG), STANDARD_CONFIG


@pytest.fixture
def sample_batch_tiny():
    """Optimized sample batch for tiny model."""
    batch_size = 1  # Reduced from 2
    return (
        torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, TINY_CONFIG.block_size)),
        torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, TINY_CONFIG.block_size)),
    )


@pytest.fixture(scope="function")
def sample_batch_small():
    """Optimized sample batch for small model."""
    batch_size = 1  # Reduced from 2
    return (
        torch.randint(
            0, SMALL_CONFIG.vocab_size, (batch_size, SMALL_CONFIG.block_size)
        ),
        torch.randint(
            0, SMALL_CONFIG.vocab_size, (batch_size, SMALL_CONFIG.block_size)
        ),
    )


@pytest.fixture(scope="function")  # Cache batch for reuse
def cached_sample_batch_tiny():
    """Cached sample batch for tiny model to avoid recreation."""
    torch.manual_seed(42)
    batch_size = 1
    return (
        torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, TINY_CONFIG.block_size)),
        torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, TINY_CONFIG.block_size)),
    )


@pytest.fixture(scope="function")  # Cache batch for reuse
def cached_sample_batch_small():
    """Cached sample batch for small model to avoid recreation."""
    torch.manual_seed(42)
    batch_size = 1
    return (
        torch.randint(
            0, SMALL_CONFIG.vocab_size, (batch_size, SMALL_CONFIG.block_size)
        ),
        torch.randint(
            0, SMALL_CONFIG.vocab_size, (batch_size, SMALL_CONFIG.block_size)
        ),
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


def assert_valid_logging(model, batch_x=None, batch_y=None):
    """Optimized common assertions for logging functionality."""
    logger = DAGLogger()

    # If batch data provided, do a forward pass first to populate activations
    if batch_x is not None and batch_y is not None:
        model.train()
        logits, loss = model(batch_x, batch_y)
        loss.backward()  # Generate gradients for logging
        model.zero_grad()  # Clean up

    logger.compute_log_statistics(model)

    # Test console logging works (but don't actually print to reduce noise)
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
    """Optimized common setup for gradient tracking tests."""
    model.train()
    logger = DAGLogger()

    batch_size = 1  # Reduced from 2
    seq_len = min(config.block_size, 4)  # Cap sequence length for speed
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


class OptimizedMockModel(torch.nn.Module):
    """Optimized mock model for testing with minimal overhead."""

    def __init__(self, vocab_size=16):  # Reduced default vocab size
        super().__init__()
        self.vocab_size = vocab_size
        self.training = True
        # Pre-allocate common tensors to avoid repeated allocation
        self._cached_logits = {}

    def forward(self, x, y=None):
        batch_size, seq_len = x.shape

        # Use cached tensors when possible to reduce allocation overhead
        cache_key = (batch_size, seq_len)
        if cache_key not in self._cached_logits:
            self._cached_logits[cache_key] = torch.randn(
                batch_size, seq_len, self.vocab_size
            )

        logits = self._cached_logits[cache_key]
        loss = torch.tensor(1.0) if y is not None else None
        return logits, loss

    def generate(self, x, max_new_tokens=1, **kwargs):
        """Return input tensor with dummy extension."""
        batch_size, seq_len = x.shape
        # Use zeros for speed instead of random
        new_tokens = torch.zeros(
            (batch_size, max_new_tokens), dtype=torch.long, device=x.device
        )
        return torch.cat([x, new_tokens], dim=1)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def to(self, device):
        return self


# Create a shared mock model instance for reuse
@pytest.fixture(scope="function")
def shared_mock_model():
    """Shared mock model to avoid repeated instantiation."""
    return OptimizedMockModel()


def mock_encode(text):
    """Optimized mock encode function for testing."""
    # Return shorter sequences for speed
    return torch.tensor([1, 2, 3], dtype=torch.long)


def mock_decode(tokens):
    """Optimized mock decode function for testing."""
    return f"decoded_{len(tokens)}_tokens"


# Fast utility functions for common test patterns
def create_minimal_batch(vocab_size=16, seq_len=3, batch_size=1):
    """Create a minimal batch for testing."""
    return (
        torch.randint(0, vocab_size, (batch_size, seq_len)),
        torch.randint(0, vocab_size, (batch_size, seq_len)),
    )


def quick_forward_pass_test(model, config, check_gradients=False):
    """Quick forward pass test with optional gradient checking."""
    batch_x, batch_y = create_minimal_batch(
        config.vocab_size, min(config.block_size, 3), 1
    )

    model.train() if check_gradients else model.eval()

    with torch.no_grad() if not check_gradients else torch.enable_grad():
        logits, loss = model(batch_x, batch_y)

        if check_gradients and loss is not None:
            loss.backward()
            # Quick gradient check
            grad_found = any(p.grad is not None for p in model.parameters())
            model.zero_grad()
            return logits, loss, grad_found

    return logits, loss


# Legacy aliases for backwards compatibility
MockModel = OptimizedMockModel
