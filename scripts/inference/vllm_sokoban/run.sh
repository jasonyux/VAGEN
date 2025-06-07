set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export WANDB_RUN_GROUP=sokoban_debug_inference

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
# This will take the last 3 parts of the path: format/sokoban/free_think
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-2 | rev | tr '/' '-')
test_path=data/sokoban-terminal-vision/test.parquet

echo "Experiment name: $EXPERIMENT_NAME"
# run python -m vagen.server.server in a tmux session first

python -m vagen.inference.run_inference \
    --wandb_path_name=sokoban-terminal-vision \
    --inference_config_path="$SCRIPT_DIR/inference_config.yaml" \
    --model_config_path="$SCRIPT_DIR/model_config.yaml" \
    --val_files_path=$test_path
