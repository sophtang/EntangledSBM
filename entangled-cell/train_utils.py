import yaml
import string
import secrets
import os

import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torchdyn.core import NeuralODE

import torch

@torch.no_grad()
def gather_local_starts(x0s, X0_pool, N, k=64):
    # for each anchor b, take its k-NN from pool, then sample N distinct
    B, G = x0s.shape
    d2 = torch.cdist(x0s, X0_pool).pow(2)             # (B, M0)
    knn_idx = d2.topk(k=min(k, X0_pool.size(0)), largest=False).indices  # (B,k)
    x0_clusters = []
    for b in range(B):
        choices = knn_idx[b]
        pick = choices[torch.randperm(choices.numel(), device=choices.device)[:N]]
        x0_clusters.append(X0_pool[pick])             # (N,G)
    return torch.stack(x0_clusters, dim=0)            # (B,N,G)

@torch.no_grad()
def make_aligned_clusters(ot_sampler, x0s, x1s, N, replace=True, k_local=128):
    
    device, dtype = x0s.device, x0s.dtype
    
    B, G = x0s.shape
    M = x1s.shape[0]
    # Use gather_local_starts to get N distinct cells for each source
    x0_clusters = gather_local_starts(x0s, x0s, N, k=k_local).to(device=device, dtype=dtype)
    x1_clusters = torch.empty((B, N, G), device=device, dtype=dtype)
    idx1 = torch.empty((B, N), device=device, dtype=torch.long)

    # Try to get a full coupling once (preferred: row-stochastic matrix P of shape (B, M))
    P = None
    if hasattr(ot_sampler, "coupling"):
        P = ot_sampler.coupling(x0s, x1s)  # expected (B, M) torch tensor
    elif hasattr(ot_sampler, "plan"):
        P = ot_sampler.plan(x0s, x1s)      # same expectation
    # If your ot_sampler only supports sampling, we’ll fall back row-by-row below.

    for b in range(B):
        x0_b = x0s[b:b+1]                  # (1, G)

        if P is not None:
            # --- Sample N targets from the row distribution P[b] ---
            probs = P[b].clamp_min(0)
            probs = probs / probs.sum().clamp_min(1e-12)
            if replace:
                j = torch.multinomial(probs, num_samples=N, replacement=True)   # (N,)
            else:
                k = min(N, (probs > 0).sum().item())
                j = torch.multinomial(probs, num_samples=k, replacement=False)
                if k < N:  # pad by repeating the last choice to keep shape
                    j = torch.cat([j, j[-1:].expand(N-k)], dim=0)
            x1_match = x1s[j]              # (N, G)
        else:
            # --- Row-wise fallback using sampler’s own sampling API ---
            # Try to ask for N pairs at once
            got = False
            if hasattr(ot_sampler, "sample_plan"):
                try:
                    # many samplers support an argument like n_pairs / k / n
                    x0_rep, x1_match = ot_sampler.sample_plan(
                        x0_b, x1s, replace=replace, n_pairs=N
                    )
                    # x0_rep: (N, G) or (1, N, G) -> squeeze if needed
                    x1_match = x1_match.view(N, G)
                    got = True
                except TypeError:
                    pass
            if not got:
                # last resort: call sample_plan N times
                xs, ys, js = [], [], []
                for _ in range(N):
                    x0_rep, x1_one = ot_sampler.sample_plan(x0_b, x1s, replace=replace)
                    # infer index by nearest neighbor for bookkeeping (optional)
                    j_hat = torch.cdist(x1_one.view(1, -1), x1s).argmin()
                    xs.append(x0_rep.view(1, G))
                    ys.append(x1_one.view(1, G))
                    js.append(j_hat.view(1))
                x1_match = torch.cat(ys, dim=0)
                j = torch.cat(js, dim=0)

        # Fill clusters (source row replicated N times)
        #x0_clusters[b] = x0_b.expand(N, G)
        x1_clusters[b] = x1_match
        idx1[b] = j

    return x0_clusters, x1_clusters, idx1


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown configuration parameter '{key}' found in the config file."
            )
        setattr(args, key, value)
    return args


def generate_group_string(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))