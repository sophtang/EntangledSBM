# Entangled Schrödinger Bridge Matching ⚛️🌟


[**Sophia Tang**](https://sophtang.github.io/), **[Yinuo Zhang](https://www.linkedin.com/in/yinuozhang98/)**, and [**Pranam Chatterjee**](https://www.chatterjeelab.com/)

![EntangledSBM](assets/entangled.gif)

This is the repository for **Entangled Schrödinger Bridge Matching** ⚛️🌟. It is partially built on [**BranchSBM repo**](https://huggingface.co/ChatterjeeLab/BranchSBM) ([Tang et al. 2025](https://arxiv.org/abs/2506.09007)) and the [**TPS-DPS repo**](https://github.com/kiyoung98/tps-dps) ([Seong et al. 2024](https://arxiv.org/abs/2405.19961)). 

Simulating trajectories of multi-particle systems on complex energy landscapes is a central task in molecular dynamics (MD) and drug discovery, but remains challenging at scale due to computationally expensive and long simulations. Flow and Schrödinger bridge matching have been used to implicitly learn joint trajectories through data snapshots. However, many systems undergo *dynamic interactions* that evolve over their trajectory and cannot be captured through static snapshots.

**EntangledSBM** solves this by learning the first- and second-order stochastic dynamics of interacting, multi-particle systems where the direction and magnitude of each particle’s path depend dynamically on the paths of the other particles.  

🌟 We formulate the **Entangled Schrödinger Bridge (EntangledSB) problem** that aims to parameterize a **bias force** that dynamically depends on the system’s positions and velocities as they evolve over time. 

🌟 To solve the EntangledSB problem, we introduce a novel parameterization of the bias force that can be conditioned, *at inference time*, on a target distribution or terminal state, enabling the generation of trajectories to **diverse target distributions.**

🌟 We minimize the divergence of the simulated path distribution from the optimal bridge distribution using a **weighted cross-entropy** objective. 

We evaluate EntangledSBM on **mapping cell cluster dynamics under drug perturbations** and **transition path sampling (TPS) of high-dimensional molecular systems.**

## Cell-State Perturbation Experiment 🧫

In this experiment, we evaluate the ability of EntangledSBM to **generate the trajectories of cell clusters under perturbation**. We demonstrate that EntanlgedSBM accurately **reconstructs perturbed cell states** and **generalizes to divergent target states not seen during training**. 

Code and instructions to reproduce our results are provided in `/entangled-cell`.

![EntangledSBM for Cell Perturbation Modelling](assets/fig-trametinib.png)

## Transition Path Sampling Experiment ⚛️

In this experiment, we evaluate the capability of EntangledSBM in **simulating molecular dynamics (MD) trajectories** given a potential energy landscape and the starting and target metastable states. We evaluate Alanine Dipeptide and three fast-folding proteins (Chignolin, Trp-cage, and BBA) and demonstrate enhanced performance against baselines for all-atom simulations. 

Code and instructions to reproduce our results are provided in `/entangled-tps`.

![EntangledSBM for Transition Path Sampling](assets/tps-full.png)

## Citation

If you find this repository helpful for your publications, please consider citing our paper:

```python
@article{tang2025entangledsbm,
  title={Entangled Schrödinger Bridge Matching},
  author={Sophia Tang and Yinuo Zhang and Pranam Chatterjee},
  journal={arXiv preprint arXiv:2511.07406},
  year={2025}
}
```

To use this repository, you agree to abide by the MIT License.