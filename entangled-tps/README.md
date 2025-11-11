# EntangledSBM for Transition Path Sampling ⚛️ 

This part of the code is for running the transition path sampling (TPS) experiments. We demonstrate that EntanlgedSBM generates physically plausible transition paths for fast-folding proteins at an **all-atom resolution**. This codebase is partially built off of the [TPS-DPS repo](https://github.com/kiyoung98/tps-dps/tree/main).

## Environment Installation
```
conda env create -f environment.yml

conda activate entangled-tps
```

## Running Experiments

All training scripts are located in `entangled-tps/scripts/`. Each script is pre-configured for a specific experiment.

**Before running experiments:**

1. Set `HOME_LOC` to the base path where EntangledSBM is located and `ENV_PATH` to the directory where your environment is downloaded in the `.bash` files in `scripts/`
2. Replace `/path/to/your/home` to the base path where EntangledSBM is located in the `.slurm` files in `scripts/`
3. Create a path `entangled-tps/results` where the simulated trajectory figures and metrics will be saved. Also, create `entangled-tps/logs` where the training logs will be saved.
4. Activate the conda environment:
```
conda activate entangled-tps
```
5. Login to wandb using `wandb login`

**Run experiment using `nohup` with the following commands:**

```
cd entangled-tps/scripts

chmod aldp.sh

nohup ./aldp.sh > aldp.log 2>&1 &
```
**Run experiment using `slurm` with the following commands:**
```
cd entangled-tps/scripts

sbatch aldp.slurm
```

Evaluation will run automatically after each sampling step and logged with wandb.

We report the following metrics in our paper:
1. Root-Mean Squared Distance **(RMSD)** of the Kabsch-aligned coordinates averaged across 64 paths.
2. Percentage of simulated trajectories that hit the target state **(THP)**
3. Highest energy transition state along the biased trajectories averaged across the trajectories that hit the target **(ETS; kJ/mol)**

## Overview of Outputs

**Training outputs are saved to experiment-specific directories:**

```
entangled-tps/results/
├── aldp_ce/
│   └── positions/         # Generated trajectory
│   └── paths.png          # Figure of simulated trajectories
│   └── policy.pt          # Model checkpoint
```

**Training logs are saved in:**
```
entangled-tps/logs/
├── <DATE>_entangled-aldp-ce.log
├── <DATE>_entangled-chignolin-ce.log
├── <DATE>_entangled-trpcage-ce.log
└── <DATE>_entangled-bba-ce.log
```