# SFT warm-up line: GSPO video teacher, from the video SFT warm-up.
# Layout: 2 trainer + 2 rollout nodes = 4 nodes x 8 GPUs (the layout of every released config).
# Sourced by scripts/train_rl.sh.
NNODES=4
export VERL_VIDEO_DECODE_CACHE=16
USE_COMMON=1
RUN_ARGS=(
  actor_rollout_ref.actor.fsdp_config.fsdp_size=16
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  "actor_rollout_ref.model.path=${CKPT_DIR}/sft_video/hf"
  actor_rollout_ref.rollout.max_model_len=22017
  actor_rollout_ref.rollout.max_num_seqs=192
  actor_rollout_ref.rollout.nnodes=2
  data.max_prompt_length=5632
  data.train_batch_size=256
  "data.train_files=[${DATA_DIR}/rl/video/data.parquet]"
  "data.val_files=[${DATA_DIR}/rl/video/data.parquet]"
  "hydra.run.dir=${CKPT_DIR}/warmup_teacher_video"
  "trainer.default_local_dir=${CKPT_DIR}/warmup_teacher_video"
  trainer.experiment_name=warmup_teacher_video
  trainer.nnodes=2
  trainer.save_freq=20
  trainer.total_epochs=2
  trainer.total_training_steps=null
  trainer.v1.separate_async.num_warmup_batches=2
)
