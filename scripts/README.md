# Scripts

This folder holds utility helpers – mainly the `container_setup.sh` script used
by [RunPod](https://runpod.io) to spin-up a fresh container, pull the repo and
launch training.

## Quick usage

```bash
bash scripts/container_setup.sh config/train_default.py --dag-depth=2
```

The script
1. clones/updates `nanoGPT-llm-extraction`,
2. installs `requirements-dev.txt`,
3. runs `python train.py <args>` and streams logs.

CI checks (`pytest`) cover the bash script's syntax and permissions so you can
trust it to run head-less. 