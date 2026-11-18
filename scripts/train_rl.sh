#!/usr/bin/env bash
# GSPO (teachers, MixRL) and MOPD with verl. Start Ray on the NNODES nodes of the run first (scripts/ray_start.sh),
# then run on node 0:
#   bash scripts/train_rl.sh teacher_image_stem   # or teacher_image_perception, teacher_video, mopd, mixrl_on_mopd (main recipe),
#                                                 # warmup_<run> (SFT warm-up line), mixrl_baseline, mixrl_from_base
# Extra hydra overrides can follow the run name. Checkpoints go to $CKPT_DIR/<run>/global_step_<N>; a run resumes from
# the latest one. Merge one with scripts/merge_rl.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="$1"; shift
source "$REPO/configs/rl/gspo_common.sh"
source "$REPO/configs/rl/$NAME.sh"
ARGS=("${RUN_ARGS[@]}")
if [ "$USE_COMMON" = 1 ]; then ARGS=("${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"); fi
echo "[train_rl] $NAME on $NNODES nodes"
exec python -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
