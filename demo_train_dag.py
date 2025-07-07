#!/usr/bin/env python
"""
demo_train_dag.py
Demonstration of how to use train_dag.py for DAG predictor pretraining.

This script shows how to:
1. Set up the environment
2. Configure DAG pretraining parameters
3. Run DAG predictor pretraining
4. Use the pretrained weights in full model training

Usage:
    python demo_train_dag.py
"""

import os
import tempfile
from pathlib import Path


def demo_dag_pretraining():
    """Demonstrate DAG predictor pretraining workflow."""

    print("=" * 60)
    print("DAG Predictor Pretraining Demo")
    print("=" * 60)

    # 1. Configuration
    print("\n1. Configuration:")
    print("   - Uses config/train_dag_default.py")
    print("   - Optimized for fast pretraining")
    print("   - Trains only DAG predictor components")
    print("   - Uses DAGStructureDataset for structure prediction")

    # 2. Basic usage
    print("\n2. Basic Usage:")
    print("   # Set your wandb API key")
    print("   export WANDB_API_KEY=your_key_here")
    print()
    print("   # Run pretraining")
    print("   python train_dag.py config/train_dag_default.py")
    print()
    print("   # With custom parameters")
    print("   python train_dag.py config/train_dag_default.py \\")
    print("     --dag_depth=4 \\")
    print("     --max_iters=1000 \\")
    print("     --learning_rate=1e-3")

    # 3. Configuration options
    print("\n3. Key Configuration Options:")
    print("   - dag_depth: Target DAG depth (default: 4)")
    print("   - max_dag_depth: Max depth for training data (default: 6)")
    print("   - min_dag_depth: Min depth for training data (default: 1)")
    print("   - learning_rate: Learning rate (default: 5e-4)")
    print("   - batch_size: Batch size (default: 16)")
    print("   - max_iters: Training iterations (default: 5000)")

    # 4. Output
    print("\n4. Training Output:")
    print("   - Checkpoints saved to checkpoints/dag_ckpt_*.pt")
    print("   - Wandb logging for loss tracking")
    print("   - Separate losses: sign_loss, log_loss, op_loss")
    print("   - Model state includes DAG predictor weights")

    # 5. Using pretrained weights
    print("\n5. Using Pretrained Weights:")
    print("   # Resume from DAG checkpoint in main training")
    print("   python train.py config/train_gpt2.py \\")
    print("     --init_from=resume \\")
    print("     --resume_from=checkpoints/dag_ckpt_final.pt")

    # 6. Architecture details
    print("\n6. Architecture Details:")
    print("   - Freezes all parameters except dag.plan_predictor")
    print("   - Predicts: initial signs, log magnitudes, operation probabilities")
    print("   - Uses negative log-likelihood loss for operations")
    print("   - MSE loss for continuous values (signs, log magnitudes)")

    # 7. Dataset details
    print("\n7. Dataset Details:")
    print("   - DAGStructureDataset generates (text, structure) pairs")
    print("   - Text: Natural language DAG computation descriptions")
    print("   - Structure: Tensors matching DAGPlanPredictor format")
    print("   - On-the-fly generation with configurable complexity")

    print("\n" + "=" * 60)
    print("Ready to start DAG predictor pretraining!")
    print("=" * 60)


def show_config_example():
    """Show example of custom configuration."""

    print("\nExample Custom Configuration:")
    print("-" * 40)

    config_example = """
# custom_dag_config.py
# Custom configuration for DAG predictor pretraining

# Project settings
name = "my_dag_pretrain"
note = "Custom DAG pretraining experiment"

# DAG dataset parameters
max_dag_depth = 8       # Harder problems
min_dag_depth = 2       # Skip trivial cases
batch_size = 32         # Larger batches
sequence_length = 512   # Longer sequences

# Model architecture
n_layer = 8             # Deeper model
n_head = 8
n_embd = 512
dag_depth = 6           # Deeper DAG

# Training
learning_rate = 3e-4
max_iters = 10000      # Longer training
weight_decay = 1e-1
grad_clip = 1.0

# Loss weights (tune for your needs)
sign_loss_weight = 1.0
log_loss_weight = 1.0  
op_loss_weight = 3.0   # Focus on operations

# Seeds
train_seed = 42
val_seed = 43
"""

    print(config_example)


if __name__ == "__main__":
    demo_dag_pretraining()
    show_config_example()

    print("\nTo run the tests:")
    print("python -m pytest tests/test_train_dag.py -v")

    print("\nTo run a quick test:")
    print(
        "WANDB_API_KEY=dummy python train_dag.py config/train_dag_default.py --max_iters=1 --eval_only=True"
    )
