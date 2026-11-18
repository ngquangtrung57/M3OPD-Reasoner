#!/usr/bin/env bash
# Start Ray on one node of an RL run, with the environment of the run (the rollout workers inherit it).
#   node 0:           bash scripts/ray_start.sh <run> head
#   every other node: bash scripts/ray_start.sh <run> <address of node 0>
# Use the same <run> as scripts/train_rl.sh, on NNODES nodes in total (see configs/rl/<run>.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$REPO/configs/rl/$1.sh"
if [ "$2" = head ]; then
  ray start --head --port 6379 --num-gpus 8
else
  ray start --address "$2:6379" --num-gpus 8
fi
