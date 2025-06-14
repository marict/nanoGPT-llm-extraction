#!/usr/bin/env bash
# Install RunPod client and set environment variables.
# Usage: bash scripts/runpod_setup.sh <RUNPOD_API_KEY>
set -e
if [ -z "$1" ]; then
  echo "Usage: $0 <RUNPOD_API_KEY>" >&2
  exit 1
fi
pip install runpod
# persist the API key
if ! grep -q RUNPOD_API_KEY ~/.bashrc 2>/dev/null; then
  echo "export RUNPOD_API_KEY=$1" >> ~/.bashrc
else
  sed -i 's/^export RUNPOD_API_KEY=.*/export RUNPOD_API_KEY=$1/' ~/.bashrc
fi
source ~/.bashrc
