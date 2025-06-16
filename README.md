# nanoGPT DAG Experiment

This repository extends [nanoGPT](https://github.com/karpathy/nanoGPT) with a differentiable
DAG module for lightweight numeric reasoning. The DAG controller attends over
previous nodes and chooses simple operations such as addition and subtraction
(see `dag_model.py`). The `DAGGPT` model runs the transformer while preserving a
copy of the input embeddings. A separate DAG stream operates on this copy and
its final node is decoded back to a numeric token.
The module implements basic ops (`add`, `multiply`, `subtract`, `divide`, `identity`) and learns to compose them via attention.
The experiment evaluates whether this reasoning layer improves performance on small arithmetic problems.

## Installation

```bash
pip install -r requirements-dev.txt
```

## Training

Run a toy training job on CPU using the default configuration:

```bash
python train.py config/train_default.py --dag_depth=4
```

Any option in `TrainConfig` can be overridden on the command line, e.g.
`--max_iters=100` or `--batch_size=4`.

## Testing

```bash
pytest
```

The tests cover the tokenizer, DAG logic and the training script.

After running inference you can decode the DAG prediction to a float using
``DAGGPT.predict_number`` together with ``NumericTokenizer``.

## Benchmark

```bash
python bench.py
```

This benchmarks a minimal model forward and backward pass.
