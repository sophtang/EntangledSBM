#!/bin/bash

HOME_LOC=/path/to/your/home/EntangledSBM
ENV_LOC=/path/to/your/envs/entangled-tps
SCRIPT_LOC=$HOME_LOC/entangled-tps
LOG_LOC=$HOME_LOC/entangled-tps/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='entangled-aldp-ce'
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train.py \
    --wandb \
    --device "cuda:0" \
    --run_name "aldp-ce" \
    --date "$DATE-aldp-ce" \
    --num_rollouts 200 \
    --num_steps 1000 \
    --use_delta_to_target \
    --objective "ce" \
    --sigma 0.1 \
    --num_samples 64 \
    --self_normalize \
    --end_temperature 300 \
    --importance_sample \
    --sigma_max 0.1 \
    --sigma_min 0.1 \
    --timestep 1 \
    --vel_conditioned >> "${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}.log" 2>&1

conda deactivate