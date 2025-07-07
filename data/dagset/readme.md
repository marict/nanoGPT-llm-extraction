# DAG Computation Dataset

A synthetic streaming dataset of DAG (Directed Acyclic Graph) computations designed to teach language models about structured reasoning and arithmetic operations in log-space.

**Key Feature**: Zero storage required - generates infinite examples on-the-fly!

## Dataset Description

Each example represents a complete DAG computation with:
- **Initial values**: Random values converted to sign and log-magnitude representation
- **Operations**: Sequence of arithmetic operations (add, subtract, multiply, divide, identity)
- **Hard-maxed selections**: Operands are selected deterministically (not soft attention)
- **Step-by-step execution**: Shows intermediate results and final output

## Example Format

```
DAG Computation (depth=3):

v0 = 5.123456 (sign=+1.0, log_mag=1.633915)

Step 1: v1 = v0 add v0
  Result: 10.246912 (sign=+1.0, log_mag=2.327078)

Step 2: v2 = v1 multiply v0
  Result: 52.475832 (sign=+1.0, log_mag=3.960993)

Step 3: v3 = v2 subtract v1
  Result: 42.228920 (sign=+1.0, log_mag=3.742278)

Final result: v3 = 42.228920
```

## Usage

### Basic Usage

```python
from data.dagset import create_dag_dataloaders

# Create streaming data loaders
train_loader, val_loader = create_dag_dataloaders(
    train_examples_per_batch=1000,  # Generate 1000 examples per batch
    val_examples_per_batch=100,     # Generate 100 examples per batch
    batch_size=32,                  # Training batch size
    block_size=1024,                # Sequence length
    max_depth=8,                    # Maximum DAG depth
    min_depth=1,                    # Minimum DAG depth
    train_seed=42,                  # Seed for reproducible training data
    val_seed=43,                    # Different seed for validation data
)

# Use directly in training loop
for inputs, targets in train_loader:
    # Your training code here
    logits = model(inputs)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    # ...
```

### Advanced Usage

```python
from data.dagset import StreamingDAGDataset

# Create custom dataset
dataset = StreamingDAGDataset(
    max_depth=5,
    min_depth=1,
    value_range=(-20.0, 20.0),  # Custom value range
    seed=123,
)

# Generate specific amounts
tokens, text = dataset.generate_batch(batch_size=100)
tokens = dataset.generate_tokens(num_tokens=10000)
```

## Configuration

Default parameters:
- **Depth range**: 1-8 steps
- **Initial values**: 1 per example
- **Value range**: -10.0 to +10.0
- **Generation rate**: ~5,000 examples/sec
- **Token rate**: ~1M+ tokens/sec

## Performance

- **Zero disk usage**: No files to store or manage
- **Infinite variety**: Different seeds = different datasets  
- **Perfect reproducibility**: Same seed = identical results
- **Memory efficient**: Generate only what you need
- **Fast generation**: ~1M+ tokens/sec generation rate
- **Training overhead**: <0.1s per batch

## Purpose

This dataset helps language models learn:
1. **Structured reasoning**: Following computational graphs step-by-step
2. **Arithmetic operations**: Understanding log-space arithmetic
3. **Variable tracking**: Managing intermediate results and references
4. **Pattern recognition**: Learning to predict computation outcomes 