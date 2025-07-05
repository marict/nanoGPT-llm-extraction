#!/usr/bin/env bash
# Persist-safe nanoGPT setup and training entry-point
# Saves logs and checkpoints in /runpod-volume so they survive POD restarts.

set -xe            # prints every command before it runs
mountpoint -q /runpod-volume || echo "/runpod-volume not mounted"

set -euo pipefail
exec 2>&1                     # merge stderr into stdout

start_time=$(date +%s)

log()   { printf '[%6ss] %s\n'  "$(( $(date +%s) - start_time ))" "$*"; }
err()   { log "ERROR: $*" >&2; }
trap 'err "failed at line $LINENO"' ERR

#---------------------------------------------------------------------------#
# workspace & repo
#---------------------------------------------------------------------------#
cd /workspace/repo
log "cwd $(pwd)"

# Repository should already be cloned by the calling script
if [[ ! -d .git ]]; then
    err "Repository not found. Expected to be in /workspace/repo"
    exit 1
fi

#---------------------------------------------------------------------------#
# system pkgs (guard against readonly images)
#---------------------------------------------------------------------------#
apt-get update || true
apt-get install -y --no-install-recommends tree

#---------------------------------------------------------------------------#
# python deps
#---------------------------------------------------------------------------#
log "installing python deps"
pip install -q -r requirements-dev.txt

#---------------------------------------------------------------------------#
# debugging env vars
#---------------------------------------------------------------------------#
export CUDA_LAUNCH_BLOCKING=1  # helpful for catching async CUDA errors
export TORCHINDUCTOR_AUTOTUNE=0 # disable autotune since it's buggy

#---------------------------------------------------------------------------#
# training
#---------------------------------------------------------------------------#
log_file="/runpod-volume/train_$(date +%Y%m%d_%H%M%S).log"
log "starting training – output -> $log_file"
python -u train.py "$@" 2>&1 | tee "$log_file"

log "done in $(( $(date +%s)-start_time ))s"

# Check if keep-alive flag was passed
if [[ "$*" == *"--keep-alive"* ]]; then
    log "keep-alive mode enabled – keeping container alive"
    tail -f /dev/null
fi
