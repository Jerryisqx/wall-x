#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

cd /mnt/data/x2robot_v2/yangping/github/jerry_git/wall-x

/mnt/data/x2robot_v2/yangping/miniforge3/bin/conda init
echo "conda init finished"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/data/x2robot_v2/yangping/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/data/x2robot_v2/yangping/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/mnt/data/x2robot_v2/yangping/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/data/x2robot_v2/yangping/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
echo "source finished"
conda activate github
# wandb login a1e24778b5714c21a610da45dba1af19ae73b0dc
which python

# export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

code_dir="/mnt/data/x2robot_v2/yangping/github/jerry_git/wall-x"
config_path="/mnt/data/x2robot_v2/yangping/github/jerry_git/wall-x/workspace/example/test/bright.yml"
# config_path="/x2robot_v2/yangping/github/wall-x/workspace/example/test/pick_waste.yml"


# Use a fixed port instead of a random one
# export PORT=$((21000 + $RANDOM % 30000))
export PORT=21432

MASTER_PORT=10239 # use 5 digits ports

export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

$LAUNCHER $SCRIPT $SCRIPT_ARGS

