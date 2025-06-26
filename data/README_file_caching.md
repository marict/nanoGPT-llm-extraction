# Dataset File Caching and Organization

This document explains the file existence checking and dataset organization features added to the data preparation system.

## Overview

The data preparation scripts now include smart file caching and dataset-specific folder organization to:

1. **Avoid unnecessary re-preparation** when files already exist locally or in RunPod storage
2. **Organize datasets in separate folders** to prevent conflicts
3. **Provide force override** when re-preparation is needed
4. **Maintain compatibility** with existing workflows
5. **Leverage RunPod persistent storage** for cross-session dataset persistence

## Dataset Organization Structure

### New Structure (Dataset-Specific Folders)

Each dataset is now organized in its own subfolder:

```
data/
├── shakespeare/
│   ├── shakespeare/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   ├── meta.pkl
│   │   └── input.txt
│   └── prepare.py
├── openwebtext/
│   ├── openwebtext/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   └── prepare.py
└── proofpile/
    ├── proofpile/
    │   ├── train.bin
    │   ├── val.bin
    │   └── meta.pkl
    └── prepare.py
```

### Legacy Compatibility

The training script (`train.py`) maintains backward compatibility:

1. **First checks** for new structure: `data/{dataset}/{dataset}/`
2. **Falls back** to legacy structure: `data/{dataset}/`
3. **Creates new structure** when preparing datasets

## File Existence Checking

### Multi-Tier Checking Strategy

The system now uses a sophisticated checking strategy:

1. **Local files first**: Check if files exist in local directory
2. **RunPod storage fallback**: If local files missing, check RunPod persistent storage
3. **Automatic restoration**: Copy files from RunPod to local if found
4. **Preparation only if needed**: Only download/tokenize if files missing everywhere

### Default Behavior

All prepare scripts now check for existing files before starting preparation:

```bash
# First run on fresh environment - downloads and prepares data
python data/shakespeare/prepare.py --subset 0.1
# Output: 🔄 Starting Shakespeare dataset preparation...

# Second run - uses existing local files
python data/shakespeare/prepare.py --subset 0.1
# Output: ✅ All required files found locally

# Fresh pod restart (local files gone) - restores from RunPod storage
python data/shakespeare/prepare.py --subset 0.1
# Output: 📁 Found all files in RunPod storage
#         📥 Copied train.bin from RunPod storage to local
#         ✅ All required files restored from RunPod storage
```

### Required Files

By default, the system checks for these files:
- `train.bin` - Training data
- `val.bin` - Validation data
- `meta.pkl` - Tokenizer metadata

### Force Re-preparation

Use the `--force` flag to bypass all file existence checking:

```bash
# Force re-preparation even if files exist locally or in RunPod
python data/shakespeare/prepare.py --subset 0.1 --force
```

## RunPod Persistent Storage Integration

### Automatic Cross-Session Persistence

When running on RunPod, the system provides automatic dataset persistence:

**Local Storage** (ephemeral - lost on pod restart):
```
/workspace/data/shakespeare/shakespeare/
├── train.bin
├── val.bin
└── meta.pkl
```

**RunPod Persistent Storage** (survives pod restarts):
```
/runpod-volume/data/shakespeare/
├── train.bin
├── val.bin
└── meta.pkl
```

### Intelligent Workflow

1. **Preparation Phase**: 
   - Save files locally for immediate use
   - Copy files to RunPod persistent storage for future sessions

2. **Restart/Fresh Pod**:
   - Check local files first (fast)
   - If missing, check RunPod storage (medium speed)
   - If found in RunPod, copy to local (much faster than re-preparation)
   - Only re-prepare if missing everywhere (slow)

### Console Output Examples

**First preparation on RunPod**:
```
🔄 Starting Shakespeare dataset preparation (subset: 0.1)
📥 Downloading Shakespeare dataset...
train has 28,146 tokens
val has 3,438 tokens
📁 Copied train.bin to RunPod storage: /runpod-volume/data/shakespeare/train.bin
📁 Copied val.bin to RunPod storage: /runpod-volume/data/shakespeare/val.bin
📁 Copied meta.pkl to RunPod storage: /runpod-volume/data/shakespeare/meta.pkl
✅ Preparation complete for shakespeare
📁 Files also saved to RunPod storage: /runpod-volume/data/shakespeare
```

**Subsequent run after pod restart**:
```
📁 Found all files in RunPod storage: /runpod-volume/data/shakespeare
📥 Copied train.bin from RunPod storage to local
📥 Copied val.bin from RunPod storage to local
📥 Copied meta.pkl from RunPod storage to local
✅ All required files restored from RunPod storage to /workspace/data/shakespeare/shakespeare
📁 Using existing files - Train: 28,146 tokens, Val: 3,438 tokens
```

## Command Line Interface

### New Parameter

All prepare scripts now support the `--force` parameter:

```bash
python data/shakespeare/prepare.py --help
# Shows: --force    Force re-preparation even if files already exist
```

### Usage Examples

```bash
# Prepare with subset and intelligent caching
python data/shakespeare/prepare.py --subset 0.01

# Force re-preparation (ignore existing files everywhere)
python data/shakespeare/prepare.py --subset 0.01 --force

# Use custom output directory
python data/shakespeare/prepare.py --data-dir /path/to/output

# OpenWebText with multiple processes and caching
python data/openwebtext/prepare.py --num-proc 4 --subset 0.001

# ProofPile with force re-preparation
python data/proofpile/prepare.py --subset 0.0001 --force
```

## Training Integration

### Automatic Data Discovery

The training script automatically discovers prepared datasets:

```python
# Starts training with existing data or prepares if missing
python train.py config/train_shakespeare.py
```

### Path Resolution

Training uses smart path resolution:

1. **Check new structure**: `data/{dataset}/{dataset}/`
2. **Check legacy structure**: `data/{dataset}/`
3. **Prepare if missing**: Creates new structure and checks RunPod storage

## Performance Benefits

### Time Savings

File existence checking provides significant time savings:

- **Shakespeare**: ~5 seconds saved on re-runs
- **OpenWebText**: ~30+ minutes saved on re-runs  
- **ProofPile**: ~10+ minutes saved on re-runs

### RunPod Storage Benefits

RunPod persistent storage provides additional benefits:

- **Pod restart resilience**: Datasets survive pod restarts
- **Cross-session efficiency**: No need to re-download large datasets
- **Storage cost optimization**: Avoid repeated downloads in cloud environments
- **Development workflow**: Faster iteration cycles

### Storage Organization

Dataset-specific folders provide:

- **Conflict prevention**: No mixing of different dataset files
- **Clear organization**: Easy to identify which files belong to which dataset
- **Parallel preparation**: Can prepare multiple datasets simultaneously
- **RunPod organization**: Clean persistent storage structure

## Implementation Details

### Enhanced DataPrep Class

```python
# Initialize with dataset-specific folder
prep = DataPrep(data_dir, dataset_name="shakespeare")

# Intelligent file checking (local + RunPod)
if not force and prep.check_existing_files():
    return prep.get_existing_token_counts()

# Get token counts (with RunPod fallback)
train_tokens, val_tokens = prep.get_existing_token_counts()
```

### Multi-Tier File Checking Logic

```python
def check_existing_files(self, required_files=None):
    """Check files locally first, then RunPod storage."""
    # 1. Check local files
    if all_files_exist_locally():
        return True
    
    # 2. Check RunPod storage for missing files
    if runpod_available() and all_missing_files_in_runpod():
        copy_from_runpod_to_local()
        return True
    
    # 3. Files missing everywhere
    return False
```

### RunPod Storage Operations

```python
def _copy_from_runpod_storage(self, runpod_dir, filenames):
    """Copy files from RunPod persistent storage to local."""
    for filename in filenames:
        shutil.copy2(runpod_dir / filename, self.data_dir / filename)
        print(f"📥 Copied {filename} from RunPod storage to local")
```

## Migration Guide

### For Existing Projects

1. **No action required** - training scripts work with both old and new structures
2. **Run prepare scripts** to create new organized structure
3. **Old files remain** - can be manually cleaned up if desired
4. **RunPod users**: Existing `/runpod-volume` data will be automatically detected

### For New Projects

- **Use new structure** automatically by running prepare scripts
- **Specify `--force`** when you need to re-prepare existing datasets
- **Leverage file caching** for faster iteration during development
- **RunPod persistence** works automatically - no configuration needed

## Testing

Comprehensive test coverage ensures reliability:

```bash
# Test all file caching functionality
python -m pytest tests/test_common_prep.py::TestDatasetSubfolders -v

# Test force parameter
python -m pytest tests/test_common_prep.py::TestForceParameter -v

# Test RunPod integration with new checking
python -m pytest tests/test_common_prep.py::TestRunPodPreparationChecking -v

# Test RunPod storage functionality
python -m pytest tests/test_common_prep.py::TestRunPodStorage -v
```

## Benefits Summary

1. **⚡ Faster Development**: Skip re-preparation when files exist locally or in RunPod
2. **🗂️ Better Organization**: Dataset-specific folders prevent conflicts  
3. **🔄 Force Override**: Re-prepare when needed with `--force`
4. **📁 RunPod Integration**: Automatic persistent storage on cloud with intelligent restoration
5. **🔄 Backward Compatibility**: Works with existing workflows
6. **🧪 Thoroughly Tested**: Comprehensive test coverage ensures reliability
7. **☁️ Cloud Optimized**: Designed for RunPod workflows with cross-session persistence
8. **💰 Cost Efficient**: Reduce repeated downloads and processing in cloud environments 