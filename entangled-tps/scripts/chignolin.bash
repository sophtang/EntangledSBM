#!/bin/bash

HOME_LOC=/path/to/your/home/EntangledSBM
ENV_LOC=/path/to/your/envs/entangled-tps
SCRIPT_LOC=$HOME_LOC/entangled-tps
LOG_LOC=$HOME_LOC/entangled-tps/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='entangled-chignolin-ce'
# set 3 have skip connection
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train.py \
    --wandb \
    --run_name "chignolin-ce" \
    --device "cuda:0" \
    --date "$DATE-chignolin-ce" \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --use_delta_to_target \
    --num_steps 5000 \
    --sigma 0.5 \
    --num_rollouts 200 \
    --batch_size 1 \
    --buffer_size 100 \
    --use_delta_to_target \
    --objective "ce" \
    --self_normalize \
    --num_samples 64 \
    --importance_sample \
    --sigma_max 1.0 \
    --sigma_min 0.5 \
    --end_temperature 300 \
    --timestep 1 \
    --vel_conditioned >> "${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}.log" 2>&1

conda deactivate