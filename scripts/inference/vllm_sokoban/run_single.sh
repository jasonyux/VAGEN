set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export WANDB_RUN_GROUP=sokoban_rollout_ablation_inference

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
# This will take the last 3 parts of the path: format/sokoban/free_think
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-2 | rev | tr '/' '-')
echo "Experiment name: $EXPERIMENT_NAME"
# run python -m vagen.server.server in a tmux session first

test_path=data/sokoban-terminal-vision/test.parquet
MODEL_PATH=checkpoints/dyna_rl/ppo_sokoban_terminal_longtraj-use_loss_maskTrue-use_gae_maskTrue/global_step_200/checkpoint-200-actor
MODEL_ID=qwen2.5-vl-3b-ppo-longtraj

python -m vagen.inference.run_single_inference \
    --wandb_path_name=sokoban-terminal-vision \
    --inference_config_path="$SCRIPT_DIR/inference_config.yaml" \
    --val_files_path=$test_path \
    --model_id=$MODEL_ID \
    --model_name=$MODEL_PATH \
    --max_tokens=2048 \
    --top_p=0.95 \
    --temperature=0.7 \
    --tensor_parallel_size=2
