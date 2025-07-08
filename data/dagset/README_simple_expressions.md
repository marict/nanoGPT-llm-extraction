# Simple Mathematical Expression Generation

This updated DAG dataset now generates simple one-liner mathematical expressions instead of verbose DAG computations. The expressions can be generated in both numeric and English word formats.

## Features

### Expression Types

The dataset generates mathematical expressions with configurable complexity:

- **2 terms, 1 operator**: `28.7 * 95.9`
- **3 terms, 2 operators**: `(52.05 * 86.217) + 38.0`
- **4 terms, 3 operators**: `41.83 * 8.28 + 2.26 * 1.2`
- **5+ terms**: `40.1 * 58.732 + 63.5 + 58.4 - 3.1`

### English Conversion

Numbers and operators can be converted to English words:

- **Numeric**: `(5.22 - 3.213) / 2.32`
- **English**: `( five point two two minus three point two one three ) divided by two point three two`

### Operators

Supports basic mathematical operations:
- Addition: `+` → "plus", "added to"  
- Subtraction: `-` → "minus", "subtract", "less"
- Multiplication: `*` → "times", "multiplied by"
- Division: `/` → "divided by", "over", "divide by"

## Usage

### Basic Usage

```python
from data.dagset.streaming import StreamingDAGDataset

# Generate numeric expressions
dataset = StreamingDAGDataset(
    max_depth=2,  # 3 terms, 2 operators
    convert_to_english=False
)

tokens, text = dataset.generate_batch(5)
print(text)
# Output: 
# 11.222 + 27.575 + 14.0
# ---
# 23.3 - 2.751 / 71.6
# ---
# ...
```

### English Conversion

```python
# Generate expressions with English conversion
english_dataset = StreamingDAGDataset(
    max_depth=2,
    convert_to_english=True,
    english_conversion_probability=0.3  # 30% conversion rate
)

tokens, text = english_dataset.generate_batch(5)
print(text)
# Output:
# eleven point two two two plus twenty-seven point five seven five plus fourteen
# ---
# 23.3 - 2.751 / 71.6  # Some remain numeric
# ---
# ...
```

### Training Data Loaders

```python
from data.dagset.streaming import create_dag_dataloaders

train_loader, val_loader = create_dag_dataloaders(
    train_examples_per_batch=1000,
    batch_size=32,
    max_depth=2,
    convert_to_english=True,
    english_conversion_probability=0.3
)

# Use with training loop
for inputs, targets in train_loader:
    # inputs.shape: [32, 1024]
    # targets.shape: [32, 1024]
    pass
```

### Train Predictor Jobs

For `train_predictor.py` jobs, English conversion is automatically enabled with a hard-set 30% rate:

```python
from data.dagset.streaming import create_dag_structure_dataloaders

# English conversion is automatically set to 30% for predictor training
train_loader, val_loader = create_dag_structure_dataloaders(
    train_batch_size=32,
    val_batch_size=32,
    max_depth=4,
    train_seed=42,
    val_seed=43
)
```

## Configuration Options

- `max_depth`: Controls expression complexity (depth=2 → 3 terms, 2 operators)
- `value_range`: Range for numeric values (default: (0.1, 100.0))
- `convert_to_english`: Enable English word conversion
- `english_conversion_probability`: Probability of conversion (0.0 to 1.0)
  - **Note**: For `train_predictor.py` jobs, English conversion is hard-set to 30% and cannot be configured

## Dependencies

- `num2words`: For number-to-words conversion (added to requirements-dev.txt)

## Example Scripts

- `example_english_conversion.py`: Basic demonstration
- `test_simple_expressions.py`: Comprehensive examples with various complexities

## Example Output

```
Numeric expressions:
- 28.7 * 95.9
- (52.05 * 86.217) + 38.0  
- 41.83 * 8.28 + 2.26 * 1.2

English expressions:
- twenty-eight point seven times ninety-five point nine
- ( fifty-two point zero five times eighty-six point two one seven ) plus thirty-eight
- forty-one point eight three times eight point two eight plus two point two six times one point two
```

This provides a rich dataset for training language models on mathematical reasoning with both symbolic and natural language representations. 