# Container Setup Scripts

This directory contains scripts for setting up and testing the container environment for nanoGPT training.

## Files

- `container_setup.sh` - Main container setup script that handles repository cloning, package installation, and training execution
- `README.md` - This file

## Usage

### Running the Container Setup Script

```bash
# Basic usage
bash scripts/container_setup.sh <training_args>

# Example with specific config and DAG depth
bash scripts/container_setup.sh config/train_default.py --dag-depth=2

# Example with custom parameters
bash scripts/container_setup.sh config/train_gpt2.py --batch_size=8 --max_iters=1000
```

### Testing the Script

```bash
# Run the Python test script
python tests/test_container_setup.py
```

## What the Container Setup Script Does

1. **Timing Setup**: Records start time for elapsed time tracking
2. **Repository Setup**: Clones or updates the nanoGPT repository
3. **System Packages**: Installs required system packages (tree)
4. **Python Dependencies**: Installs Python requirements from requirements-dev.txt
5. **Training Execution**: Runs the training script with provided arguments
6. **Logging**: Saves all output to a timestamped log file

## Output Format

The script provides timing information for each step:

```
[0s] Starting container setup
[1s] Cloning repository
[4s] Repository setup completed
[5s] Installing system packages
[14s] System packages installed
[15s] Installing Python dependencies
[44s] Python dependencies installed
[45s] Starting training
[544s] Training completed
```

## Integration with RunPod

The container setup script is automatically downloaded and executed by the RunPod service. The Python code in `runpod_service.py` downloads this script from the GitHub repository and runs it with the appropriate training arguments.

## Testing

The Python test script (`tests/test_container_setup.py`) validates:

- ✅ Script file existence and permissions
- ✅ Bash syntax validity
- ✅ Required commands and patterns
- ✅ Error handling setup
- ✅ Timing logic
- ✅ Argument handling
- ✅ Script structure and flow

Run tests before deploying to ensure the script is working correctly:

```bash
python tests/test_container_setup.py
``` 