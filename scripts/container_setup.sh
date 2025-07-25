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
# system pkgs (robust apt update – NVIDIA repo can have sync issues)
#---------------------------------------------------------------------------#

# Remove any NVIDIA/CUDA apt sources unconditionally (we don't need them)
log "removing NVIDIA/CUDA apt sources"
# 1) Drop any source-list files that reference NVIDIA or CUDA
find /etc/apt/sources.list.d -type f \( -iname "*nvidia*" -o -iname "*cuda*" \) -exec rm -f {} + || true

# 2) Remove any fragment files that *reference* NVIDIA/CUDA, regardless of their filename.
#    For the main sources.list we strip offending lines in-place; for others we delete the file.

#    Handle the main sources.list separately (cannot delete entirely).
if grep -qiE "(nvidia|cuda)" /etc/apt/sources.list; then
    log "stripping NVIDIA/CUDA lines from /etc/apt/sources.list"
    grep -viE "(nvidia|cuda)" /etc/apt/sources.list > /etc/apt/sources.list.clean
    mv /etc/apt/sources.list.clean /etc/apt/sources.list
fi

#    Now iterate over every source fragment file in sources.list.d and drop the file if it references NVIDIA/CUDA.
for src_file in /etc/apt/sources.list.d/*; do
    if [[ -f "${src_file}" ]] && grep -qiE "(nvidia|cuda)" "${src_file}"; then
        log "removing source file ${src_file} that references NVIDIA/CUDA"
        rm -f "${src_file}" || true
    fi
done

# 3) Purge any cached package lists that still reference NVIDIA/CUDA to avoid stale fetches.
rm -f /var/lib/apt/lists/*nvidia* /var/lib/apt/lists/*cuda* || true

apt-get clean

# Refresh package lists once (should succeed now)
log "updating apt repositories"
apt-get update -y || true

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
python -u "$@" 2>&1 | tee "$log_file"

log "done in $(( $(date +%s)-start_time ))s"

# Check if keep-alive flag was passed
if [[ "$*" == *"--keep-alive"* ]]; then
    log "keep-alive mode enabled – keeping container alive"
    tail -f /dev/null
fi
