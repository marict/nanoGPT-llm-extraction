#!/bin/bash

# Enable error handling and logging
set -e
exec 2>&1  # Redirect stderr to stdout so we capture all output

# Container setup script for nanoGPT training
# This script handles the complete setup and training process

echo "=== CONTAINER SETUP STARTING ==="
echo "Script version: $(date)"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "Environment:"
env | sort
echo "=== END ENVIRONMENT ==="

start_time=$(date +%s)
echo "[0s] Starting container setup"

# Function to log errors and continue
log_error() {
    echo "ERROR: $1" >&2
    echo "[$(($(date +%s) - start_time))s] ERROR: $1"
}

# Function to log info
log_info() {
    echo "[$(($(date +%s) - start_time))s] $1"
}

# Trap to catch errors and log them
trap 'log_error "Script failed at line $LINENO"' ERR

echo "=== WORKSPACE SETUP ==="
cd /workspace
log_info "Changed to /workspace directory"
log_info "Current directory: $(pwd)"
log_info "Directory contents:"
ls -la

echo "=== REPOSITORY SETUP ==="
log_info "Cloning repository"

# Clone or update the repository with error handling
if [ -d repo ]; then
    log_info "Repository exists, pulling latest changes"
    cd repo
    git status || log_error "git status failed"
    git pull || log_error "git pull failed"
else
    log_info "Repository does not exist, cloning"
    git clone https://github.com/marict/nanoGPT-llm-extraction.git repo || log_error "git clone failed"
    cd repo
fi

log_info "Repository setup completed"
log_info "Current directory: $(pwd)"
log_info "Repository contents:"
ls -la

echo "=== SYSTEM PACKAGES ==="
log_info "Installing system packages"
apt-get update || log_error "apt-get update failed"
apt-get install -y tree || log_error "apt-get install tree failed"

log_info "System packages installed"

echo "=== DIRECTORY STRUCTURE ==="
echo "=== Directory Structure ==="
tree || log_error "tree command failed"
echo "=== Current Directory ==="
pwd

echo "=== PYTHON DEPENDENCIES ==="
log_info "Installing Python dependencies"

# Check if requirements file exists
if [ ! -f "requirements-dev.txt" ]; then
    log_error "requirements-dev.txt not found"
    echo "Available files:"
    ls -la *.txt || true
    exit 1
fi

log_info "Requirements file found, installing dependencies"
pip install -q -r requirements-dev.txt || log_error "pip install failed"

log_info "Python dependencies installed"

echo "=== TRAINING SETUP ==="
log_info "Starting training"

# Log the command we're about to run
echo "Training command: python train.py $@"
echo "Arguments received: $@"

# Execute the training command passed as arguments
python train.py "$@" 2>&1 | tee /workspace/train_$(date +%Y%m%d_%H%M%S).log || {
    log_error "Training failed with exit code $?"
    echo "Last 50 lines of log:"
    tail -50 /workspace/train_$(date +%Y%m%d_%H%M%S).log || true
    exit 1
}

log_info "Training completed"

echo "=== CONTAINER SETUP COMPLETED ==="
echo "Total time: $(($(date +%s) - start_time))s"

# Keep container running
echo "Keeping container alive..."
tail -f /dev/null 