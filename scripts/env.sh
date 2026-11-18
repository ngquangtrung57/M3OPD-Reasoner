# Environment of our training runs. Sourced by the train scripts.
# Set DATA_DIR (output of m3opd.data.prepare), CKPT_DIR (checkpoints and merged models), BASE_MODEL (Qwen/Qwen3.5-9B
# local dir) and THIRD_PARTY (default: <repo>/third_party, created by scripts/setup_third_party.sh) before you start.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
: "${DATA_DIR:?set DATA_DIR}" "${CKPT_DIR:?set CKPT_DIR}" "${BASE_MODEL:?set BASE_MODEL}"
export DATA_DIR CKPT_DIR BASE_MODEL
export THIRD_PARTY="${THIRD_PARTY:-$REPO/third_party}"
export PYTHONPATH="$REPO:$THIRD_PARTY/verl:$THIRD_PARTY/lmms-engine/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export FORCE_QWENVL_VIDEO_READER=torchcodec   # same video reader in SFT, rollout and teacher scoring
# RL (verl + vLLM)
export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=600000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export VERL_VIDEO_DECODE_THREADS=1
