import logging
import sys

import torch
import wandb

from .plot_color import Plot
from .metrics import Metric

class Logger:
    def __init__(self, args, mds):
        self.molecule = args.molecule
        self.save_dir = args.save_dir
        self.wandb = args.wandb
        self.plot = Plot(args, mds)
        self.metric = Metric(args, mds)
        self.rmsd = float("inf")

    def __call__(self, loss, rollout, policy):
        metrics = self.metric()
        if self.rmsd > metrics["rmsd"]:
            self.rmsd = metrics["rmsd"]
            torch.save(policy.state_dict(), f"{self.save_dir}/policy.pt")
        if self.wandb:
            if metrics["ets"] is not None:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "rmsd_std": metrics["rmsd_std"],
                    "thp": metrics["thp"],
                    "ets": metrics["ets"],
                    "ets_std": metrics["ets_std"],
                    "loss": loss
                })
            else:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "rmsd_std": metrics["rmsd_std"],
                    "thp": metrics["thp"],
                    "loss": loss
                })

