#!/usr/bin/env bash
# Merge an lmms-engine checkpoint to a Hugging Face model in $CKPT_DIR/<run>/hf (the input of the next stage):
#   bash scripts/merge_sft.sh <run> <step>
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK="$CKPT_DIR/$1/checkpoint-$2"; OUT="$CKPT_DIR/$1/hf"
python -m lmms_engine.merger --checkpoint_path "$CK" --output_path "$OUT" --model_general_type image_text_to_text
python -m m3opd.ckpt "$OUT" --base "$BASE_MODEL"
