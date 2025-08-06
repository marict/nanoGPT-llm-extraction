# NALM Dataset

This module implements the **Single Module Arithmetic Task** dataset from the NALM (Neural Arithmetic Logic Modules) benchmark, as described in "A Primer for Neural Arithmetic Logic Modules" (Mistry et al., 2022).

## Overview

The NALM dataset generates synthetic arithmetic expressions for training and evaluating neural networks on basic mathematical operations. It's designed to test:

- **Interpolation**: How well models perform on values within the training range
- **Extrapolation**: How well models generalize to values outside the training range
- **Mathematical reasoning**: Basic arithmetic operations (addition, subtraction, multiplication, division)

## Features

- **Streaming dataset**: Generates examples on-the-fly, no disk storage required
- **Configurable operations**: Support for addition, subtraction, multiplication, and division
- **Range control**: Separate ranges for training, validation, and extrapolation testing
- **Reproducible**: Deterministic generation with configurable seeds
- **Tokenized**: Uses GPT-2 tokenizer for language model training

## Usage

### Basic Usage

```python
from data.nalm.streaming import create_nalm_dataloaders

# Create dataloaders
train_dataloader, val_dataloader = create_nalm_dataloaders(
    train_range=(-10.0, 10.0),
    val_range=(-10.0, 10.0),
    extrapolation_range=(-100.0, 100.0),
    operations=["add", "sub", "mul", "div"],
    batch_size=32,
    train_examples=10000,
    val_examples=1000,
    seed=42,
)

# Use in training loop
for batch in train_dataloader:
    tokens = batch["tokens"]  # Tokenized expressions
    results = batch["result"]  # Target results
    expressions = batch["expression"]  # Human-readable expressions
    # ... training code
```

### Dataset Preparation

```python
from data import prepare_dataset

# Prepare the dataset (creates metadata files)
train_tokens, val_tokens = prepare_dataset(
    dataset="nalm",
    data_dir=Path("data"),
    subset=1.0,
    force=False,
)
```

### Training Script

Use the provided training script:

```bash
python train_nalm.py
```

Or use the main training script with NALM dataset:

```bash
python train.py --dataset nalm --config config/train_nalm.py
```

## Configuration

The dataset supports various configuration options:

### Operations
- `"add"`: Addition (a + b)
- `"sub"`: Subtraction (a - b)  
- `"mul"`: Multiplication (a * b)
- `"div"`: Division (a / b)

### Ranges
- `train_range`: Range for training data generation
- `val_range`: Range for validation data generation
- `extrapolation_range`: Range for testing generalization

### Example Configuration

```python
nalm_config = {
    "train_range": (-10.0, 10.0),      # Training on [-10, 10]
    "val_range": (-10.0, 10.0),        # Validation on [-10, 10]  
    "extrapolation_range": (-100.0, 100.0),  # Test on [-100, 100]
    "operations": ["add", "sub", "mul", "div"],
    "batch_size": 32,
    "train_examples": 10000,
    "val_examples": 1000,
    "seed": 42,
}
```

## Data Format

Each example contains:

```python
{
    "tokens": torch.Tensor,      # Tokenized expression (GPT-2 tokens)
    "expression": str,           # Human-readable expression (e.g., "3.14159 + 2.71828 = 5.85987")
    "a": float,                  # First operand
    "b": float,                  # Second operand  
    "operation": str,            # Operation type ("add", "sub", "mul", "div")
    "result": float,             # Target result
}
```

## Evaluation

The dataset includes evaluation functions:

```python
from data.nalm.streaming import evaluate_nalm_model

metrics = evaluate_nalm_model(model, val_dataloader, device="cpu")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"MSE: {metrics['mse']:.6f}")
```

## Integration with DAG-GPT

The NALM dataset can be used to benchmark your DAG-GPT model:

1. **Standard GPT**: Train a regular GPT model on NALM
2. **DAG-GPT**: Train your DAG-integrated model on NALM
3. **Comparison**: Compare performance on interpolation vs extrapolation

### Example DAG-GPT Training

```python
# In config/train_nalm.py
use_dag = True
dag_config = {
    "dag_depth": 4,
    "max_digits": 8,
    "max_decimal_places": 4,
}
```

## Testing

Run the tests to verify the dataset works correctly:

```bash
python -m pytest tests/test_nalm_dataset.py -v
```

## References

- Mistry, J., et al. (2022). "A Primer for Neural Arithmetic Logic Modules". arXiv preprint arXiv:2206.08452.
- NALM Benchmark Repository: [GitHub](https://github.com/nalm-org/nalm-benchmark)

## Notes

- The dataset is **streaming-based**, meaning it generates infinite examples
- **Deterministic generation** ensures reproducible results with fixed seeds
- **Extrapolation testing** uses different ranges to test generalization
- **Division by zero** is handled automatically by adjusting small denominators
- **Tokenization** uses GPT-2 tokenizer for compatibility with language models 