#!/bin/bash
set -e

# Container setup script for nanoGPT training
# This script handles the complete setup and training process

start_time=$(date +%s)
echo "[0s] Starting container setup"

cd /workspace
echo "[$(($(date +%s) - start_time))s] Cloning repository"

# Clone or update the repository
if [ -d repo ]; then
    git -C repo pull
else
    git clone https://github.com/marict/nanoGPT-llm-extraction.git repo
fi

echo "[$(($(date +%s) - start_time))s] Repository setup completed"
cd repo

echo "[$(($(date +%s) - start_time))s] Installing system packages"
apt-get update && apt-get install -y tree

echo "[$(($(date +%s) - start_time))s] System packages installed"

echo "=== Directory Structure ==="
tree
echo "=== Current Directory ==="
pwd

echo "[$(($(date +%s) - start_time))s] Installing Python dependencies"
pip install -q -r requirements-dev.txt

echo "[$(($(date +%s) - start_time))s] Python dependencies installed"

echo "[$(($(date +%s) - start_time))s] Starting training"

# Execute the training command passed as arguments
python train.py "$@" 2>&1 | tee /workspace/train_$(date +%Y%m%d_%H%M%S).log

echo "[$(($(date +%s) - start_time))s] Training completed"

# Keep container running
tail -f /dev/null 