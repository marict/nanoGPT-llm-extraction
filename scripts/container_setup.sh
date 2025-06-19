#!/usr/bin/env bash
# Persist-safe nanoGPT setup and training entry-point
# Saves logs and checkpoints in /runpod-volume so they survive POD restarts.

set -euo pipefail
exec 2>&1                     # merge stderr into stdout

start_time=$(date +%s)

log()   { printf '[%6ss] %s\n'  "$(( $(date +%s) - start_time ))" "$*"; }
err()   { log "ERROR: $*" >&2; }
trap 'err "failed at line $LINENO"' ERR

#---------------------------------------------------------------------------#
# workspace & repo
#---------------------------------------------------------------------------#
cd /runpod-volume
log "cwd $(pwd)"

REPO_URL="https://github.com/marict/nanoGPT-llm-extraction.git"

if [[ -d repo/.git ]]; then
    log "repo exists – git pull"
    git -C repo pull
else
    log "cloning repo"
    git clone "$REPO_URL" repo
fi

#---------------------------------------------------------------------------#
# system pkgs (guard against readonly images)
#---------------------------------------------------------------------------#
apt-get update || true
apt-get install -y --no-install-recommends tree

#---------------------------------------------------------------------------#
# python deps
#---------------------------------------------------------------------------#
cd repo
log "installing python deps"
pip install -q -r requirements-dev.txt

#---------------------------------------------------------------------------#
# training
#---------------------------------------------------------------------------#
log_file="/runpod-volume/train_$(date +%Y%m%d_%H%M%S).log"
log "starting training – output -> $log_file"
python train.py "$@" 2>&1 | tee "$log_file"

log "done in $(( $(date +%s)-start_time ))s – keeping container alive"
tail -f /dev/null
