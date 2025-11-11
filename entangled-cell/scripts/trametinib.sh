#!/bin/bash

HOME_LOC=/path/to/your/home/EntangledSBM
ENV_LOC=/path/to/your/envs/entangled-cell
SCRIPT_LOC=$HOME_LOC/entangled-cell
LOG_LOC=$HOME_LOC/entangled-cell/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='trametinib-ce-50D'
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train_cell.py \
    --config_path "$SCRIPT_LOC/configs/trametinib_50D.yaml" \
    --save_dir "$SCRIPT_LOC/results/$DATE-trametinib-ce-50D" \
    --root_dir "$SCRIPT_LOC" \
    --wandb \
    --device "cuda:0" \
    --run_name "trametinib-ce-50D" \
    --date "$DATE-trametinib-ce-50D" \
    --num_rollouts 100 \
    --num_steps 100 \
    --use_delta_to_target \
    --objective "ce" \
    --sigma 0.1 \
    --num_samples 64 \
    --self_normalize \
    --kT 0.1 \
    --unseen \
    --num_particles 16 \
    --batch_size 64 \
    --friction 2.0 \
    --vel_conditioned >> ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_train.log 2>&1

conda deactivate