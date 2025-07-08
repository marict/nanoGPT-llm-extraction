# Dataset Organization and Caching

The data preparation system automatically organizes datasets and caches files to avoid unnecessary re-preparation.

## Dataset Organization

Each dataset is organized in its own subfolder:

```
data/
├── shakespeare/
│   ├── shakespeare/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
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

## Automatic Caching

The system automatically checks for existing files before preparing datasets:

1. **Local files first**: Check if files exist locally
2. **RunPod storage**: If on RunPod, check persistent storage (`/runpod-volume/`)
3. **Auto-restore**: Copy files from RunPod storage to local if found
4. **Prepare only if needed**: Only download/tokenize if files are missing

## Usage

```bash
# First run - prepares data
python data/shakespeare/prepare.py --subset 0.1

# Second run - uses cached files
python data/shakespeare/prepare.py --subset 0.1
# Output: ✅ All required files found locally

# Force re-preparation
python data/shakespeare/prepare.py --subset 0.1 --force
```

## RunPod Integration

On RunPod, datasets are automatically saved to persistent storage and restored across pod restarts:

- **Local**: `/workspace/data/shakespeare/shakespeare/` (ephemeral)
- **Persistent**: `/runpod-volume/data/shakespeare/` (survives restarts)

The system handles copying between local and persistent storage automatically. 