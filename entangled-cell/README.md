# EntangledSBM for Cell Perturbation Modelling 🧫

This part of the code is for running the cell perturbation experiments. We demonstrate that EntanlgedSBM accurately **reconstructs perturbed cell states** and **generalizes to divergent target states not seen during training**. 

## Environment Installation
```
conda env create -f environment.yml

conda activate entangled-cell
```

## Data

We use perturbation data from the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M) containing control DMSO-treated cell data and perturbed cell data. 

The raw data contains a total of 60K genes. We select the top 2000 highly variable genes (HVGs) and perform principal component analysis (PCA), to maximally capture the variance in the data via the top principal components (38% in the top-50 PCs). **Our goal is to learn the dynamic trajectories that map control cell clusters to the perturbd cell clusters.**

**Specifically, we model the following perturbations**:

1. **Clonidine**: Cell states under 5uM Clonidine perturbation at various PC dimensions (50D, 100D, 150D) with 1 unseen population.
2. **Trametinib**: Cell states under 5uM Trametinib perturbation (50D) with 2 unseen populations.

Processed data files are stored in:
```
entangled-cell/data/
├── pca_and_leiden_labels.csv              # Clonidine data
└── Trametinib_5.0uM_pca_and_leidenumap_labels.csv  # Trametinib data
```

## Running Experiments

All training scripts are located in `entangled-cell/scripts/`. Each script is pre-configured for a specific experiment.

**Before running experiments:**

1. Set `HOME_LOC` to the base path where TR2-D2 is located and `ENV_PATH` to the directory where your environment is downloaded in the `.sh` files in `scripts/`
2. Create a path `entangled-cell/results` where the simulated trajectory figures and metrics will be saved. Also, create `entangled-cell/logs` where the training logs will be saved.
3. Activate the conda environment:
```
conda activate entangled-cell
```
4. Login to wandb using `wandb login`

**Run experiment using `nohup` with the following commands:**

```
cd entangled-cell/scripts

chmod clonidine50.sh

nohup ./clonidine50.sh > clonidine50.log 2>&1 &
```
Evaluation will run automatically after the specified number of rollouts `--num_rollouts` is finished. To see metrics, go to `results/<experiment>/metrics/` or the end of `logs/<experiment>.log`. 

For Clonidine, `x1_1` indicates the cell cluster that is sampled from for training and `x1_2` is the held-out cell cluster. For Trametinib `x1_1` indicates the cell cluster that is sampled from for training and `x1_2` and `x1_3` are the held-out cell clusters.

We report the following metrics for each of the clusters in our paper:
1. Maximum Mean Discrepancy (RBF-MMD) of simualted cell cluster with target cell cluster (same cell count).
2. 1-Wasserstein and 2-Wasserstein distances against full cell population in the cluster.

## Overview of Outputs

**Training outputs are saved to experiment-specific directories:**

```
entangled-cell/results/
├── clonidine_ce_50D/
│   └── positions/         # Generated trajectory
│   └── metrics/         # JSON of metrics
│   └── figures/         # Figures of simulated trajectories
```

**PyTorch Lightning automatically saves model checkpoints to:**

```
entangled-cell/scripts/lightning_logs/
├── <wandb-run-id>/
│   ├── checkpoints/
│   │   ├── epoch=N-step=M.ckpt     # Checkpoint 
```

**Training logs are saved in:**
```
entangled-cell/logs/
├── <DATE>_clonidine-ce-50D_train.log
├── <DATE>_clonidine-ce-100D_train.log
├── <DATE>_clonidine-ce-150D_train.log
└── <DATE>_trametinib-ce-50D_train.log
```