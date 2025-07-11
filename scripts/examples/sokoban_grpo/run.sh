set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

export CUDA_VISIBLE_DEVICES=4,5,6,7


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# run python -m vagen.server.server in a tmux session first
# python -m vagen.env.create_dataset \
#     --yaml_path "$SCRIPT_DIR/env_config.yaml" \
#     --train_path "data/sokoban-vision-debug/train.parquet" \
#     --test_path "data/sokoban-vision-debug/test.parquet" \

# max_trajectory_length = max_prompt_length + max_response_length

use_multi_turn_reward=True  # default: False
exp_name=grpo_sokoban_vision-mtreward$use_multi_turn_reward

rm -f logs/$exp_name.log

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/sokoban-vision-debug/train.parquet \
    data.val_files=data/sokoban-vision-debug/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.max_trajectory_length=2048 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='dyna_rl' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=70 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    rollout_manager.max_turns=3 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=$use_multi_turn_reward \
    rollout_manager.use_loss_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=8 \
    2>&1 | tee logs/$exp_name.log
