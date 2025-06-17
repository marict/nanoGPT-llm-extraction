#!/usr/bin/env bash
# Install dependencies and set environment variables for Lambda Cloud.
# Usage: bash scripts/lambda_setup.sh <LAMBDA_API_KEY> <SSH_KEY_NAME>
set -e
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <LAMBDA_API_KEY> <SSH_KEY_NAME>" >&2
  exit 1
fi
pip install requests
# persist the API key and ssh key name
if ! grep -q LAMBDA_API_KEY ~/.bashrc 2>/dev/null; then
  echo "export LAMBDA_API_KEY=$1" >> ~/.bashrc
else
  sed -i 's/^export LAMBDA_API_KEY=.*/export LAMBDA_API_KEY=$1/' ~/.bashrc
fi
if ! grep -q LAMBDA_SSH_KEY ~/.bashrc 2>/dev/null; then
  echo "export LAMBDA_SSH_KEY=$2" >> ~/.bashrc
else
  sed -i 's/^export LAMBDA_SSH_KEY=.*/export LAMBDA_SSH_KEY=$2/' ~/.bashrc
fi
source ~/.bashrc
