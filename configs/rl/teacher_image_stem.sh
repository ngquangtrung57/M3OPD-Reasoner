# GSPO image STEM teacher, directly from Qwen3.5-9B (no SFT warm-up).
# Layout: 2 trainer + 2 rollout nodes = 4 nodes x 8 GPUs (the layout of every released config).
# Sourced by scripts/train_rl.sh.
NNODES=4
export VERL_VIDEO_DECODE_CACHE=64
USE_COMMON=1
RUN_ARGS=(
  actor_rollout_ref.actor.fsdp_config.fsdp_size=16
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  "actor_rollout_ref.model.path=${BASE_MODEL}"
  actor_rollout_ref.rollout.max_model_len=18945
  actor_rollout_ref.rollout.max_num_seqs=256
  actor_rollout_ref.rollout.nnodes=2
  data.max_prompt_length=2560
  data.train_batch_size=256
  "data.train_files=[${DATA_DIR}/rl/image_stem/data.parquet]"
  "data.val_files=[${DATA_DIR}/rl/image_stem/data.parquet]"
  "hydra.run.dir=${CKPT_DIR}/teacher_image_stem"
  "trainer.default_local_dir=${CKPT_DIR}/teacher_image_stem"
  trainer.experiment_name=teacher_image_stem
  trainer.nnodes=2
  trainer.save_freq=5
  trainer.total_epochs=3
  trainer.total_training_steps=200
  trainer.v1.separate_async.num_warmup_batches=2
)
