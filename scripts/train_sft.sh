#!/usr/bin/env bash
# SFT with lmms-engine (FSDP2). Run the same command on every node:
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0> MASTER_PORT=29500 bash scripts/train_sft.sh configs/sft/sft_image_stem.yaml
# Checkpoints go to $CKPT_DIR/<config name>/checkpoint-<step>; merge one with scripts/merge_sft.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG="$1"; NAME="$(basename "$CFG" .yaml)"
export OUTPUT_DIR="$CKPT_DIR/$NAME" OMP_NUM_THREADS=8
mkdir -p "$OUTPUT_DIR"
python -m m3opd.sft_config "$CFG" "$OUTPUT_DIR/lmms_engine.yaml"
exec python -m torch.distributed.run --nnodes "${NNODES:-4}" --node_rank "${NODE_RANK:-0}" --nproc_per_node 8 \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
  -m lmms_engine.launch.cli config_yaml="$OUTPUT_DIR/lmms_engine.yaml"
