# MixRL on top of MOPD (M3OPD-Reasoner-9B): GSPO on the prompt union, from the MOPD student.
# Layout: 2 trainer + 2 rollout nodes = 4 nodes x 8 GPUs (the layout of every released config).
# Sourced by scripts/train_rl.sh.
NNODES=4
export VERL_VIDEO_DECODE_CACHE=16
USE_COMMON=1
RUN_ARGS=(
  actor_rollout_ref.actor.fsdp_config.fsdp_size=16
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  "actor_rollout_ref.model.path=${CKPT_DIR}/mopd/hf"
  actor_rollout_ref.rollout.max_model_len=22017
  actor_rollout_ref.rollout.max_num_seqs=192
  actor_rollout_ref.rollout.nnodes=2
  data.max_prompt_length=5632
  data.train_batch_size=256
  "data.train_files=[${DATA_DIR}/rl/image_stem/data.parquet,${DATA_DIR}/rl/image_perception/data.parquet,${DATA_DIR}/rl/video/data.parquet]"
  "data.val_files=[${DATA_DIR}/rl/image_stem/data.parquet,${DATA_DIR}/rl/image_perception/data.parquet,${DATA_DIR}/rl/video/data.parquet]"
  "hydra.run.dir=${CKPT_DIR}/mixrl_on_mopd"
  "trainer.default_local_dir=${CKPT_DIR}/mixrl_on_mopd"
  trainer.experiment_name=mixrl_on_mopd
  trainer.nnodes=2
  trainer.save_freq=5
  trainer.total_epochs=3
  trainer.total_training_steps=400
  trainer.v1.separate_async.num_warmup_batches=2
)
