#!/usr/bin/env bash
# Merge a verl checkpoint to a Hugging Face model in $CKPT_DIR/<run>/hf (the input of the next stage):
#   bash scripts/merge_rl.sh <run> <step>
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK="$CKPT_DIR/$1/global_step_$2/actor"; OUT="$CKPT_DIR/$1/hf"
python -m verl.model_merger merge --backend fsdp --local_dir "$CK" --target_dir "$OUT"
python -m m3opd.ckpt "$OUT" --base "$BASE_MODEL"
