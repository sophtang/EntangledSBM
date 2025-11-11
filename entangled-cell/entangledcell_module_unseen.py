import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import numpy as np

from torch.distributions import Normal
from geo_metrics.metric_factory import natural_gradient_force
import math
from train_utils import make_aligned_clusters
from matplotlib.colors import LinearSegmentedColormap
from eval import compute_distribution_distances, compute_wasserstein_distances
import json, time, csv


class EntangledNetTrainBaseUnseen(pl.LightningModule):
    def __init__(
        self,
        args,
        bias_net, # input bias net
        data_manifold_metric,
        timepoint_data,
        ot_sampler=None,
        vel_conditioned=False,
    ):
        super().__init__()
        self.args = args
                
        self.ot_sampler = ot_sampler
        
        self.bias_net = bias_net
        
        self.data_manifold_metric = data_manifold_metric
        
        self.target_measure = PathObjective(args)
        if args.training:
            self.replay = ReplayBuffer(args)
        
        self.dt = float(1.0 / args.num_steps)
        self.std = (2.0 * args.kT / (args.friction * self.dt)) ** 0.5
        self.log_prob = Normal(0, self.std).log_prob
        self.timepoint_data = timepoint_data
        self.vel_conditioned = vel_conditioned
        self.dir_only = getattr(args, "dir_only", False)
        #self.device = args.device
        
    # returns the bias force given the position, velocity and target
    def forward(self, pos, vel, target):
        if self.vel_conditioned:
            if self.dir_only:
                velocity_magnitude = torch.norm(vel, dim=-1, keepdim=True)
                velocity_direction = vel / (velocity_magnitude + 1e-8)
                return self.bias_net(pos, velocity_direction, target)
            else:
                return self.bias_net(pos, vel, target)

        return self.bias_net(pos, target)

    def on_train_epoch_start(self):
        pass
    
    def _sample(self, x0, x1, metric_samples):
        """
        Simulate first-order velocity dynamics 
        
        x0: initial positions of batch (B, N, g)
        x1: final positions of batch (B, N, g)
        """
        device = x0.device
        print(device)
        
        B, N, G = x0.shape
        T = self.args.num_steps
        
        gamma = float(self.args.friction)
        kT = float(getattr(self.args, "kT", 0.0))

        positions = torch.empty((B, T+1, N, G), dtype=x0.dtype, device=device)
        forces = torch.empty((B, T, N, G), dtype=x0.dtype, device=device)
        biases = torch.empty((B, T, N, G), dtype=x0.dtype, device=device)

        # set initial positions
        positions[:, 0] = x0
        pos = x0.clone()
        
        target_positions = x1.clone().to(device, dtype=x0.dtype) 
        #target_positions = x1.clone().unsqueeze(dim=1).repeat(1, T+1, 1, 1)
        
        for t_idx in range(T):
            if t_idx == 0: 
                velocity = torch.zeros_like(pos)
            else:
                velocity = (pos - positions[:, t_idx-1]) / self.dt
                
            
            F_nat, _ = natural_gradient_force(self.data_manifold_metric, 
                                           pos,
                                           metric_samples,
                                           t_idx,
                                           )
            
            base_force =  F_nat # unbiased base force
            # learned bias force
            if self.vel_conditioned:
                if self.dir_only:
                    velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
                    velocity_direction = velocity / (velocity_magnitude + 1e-8)  # Avoid division by zero
                    
                    bias_force = self.bias_net(pos.detach(), 
                                            velocity_direction.detach(), 
                                            target_positions.detach()).detach()
                else:
                    bias_force = self.bias_net(pos.detach(), 
                                            velocity.detach(), 
                                            target_positions.detach()).detach()
            else:
                bias_force = self.bias_net(pos.detach(), 
                                           target_positions.detach()).detach()

            if kT > 0:
                xi = torch.randn_like(pos) * ((2.0 * kT * self.dt / gamma) ** 0.5)
            else:
                xi = 0.0
            
            pos = pos + (self.dt / gamma) * (base_force + bias_force) + xi
            
            positions[:, t_idx + 1] = pos.clone()
            forces[:, t_idx] = base_force.clone()
            biases[:, t_idx] = bias_force.clone()
        
        log_tpm, final_idx, log_ri = self.target_measure(positions, target_positions, forces)
        
        if self.args.training:
            self.replay.add_ranked((positions.detach(), 
                                    target_positions.detach(),
                                    forces.detach(),
                                    log_tpm.detach(),
                                    log_ri.detach()))
            
        for i in range(B):
            end = int(final_idx[i].item()) + 1 if torch.is_tensor(final_idx) else T + 1
            np.save(f"{self.args.save_dir}/positions/{i}.npy", positions[i, :end].detach().cpu().numpy())
        
        return positions, target_positions, forces, log_tpm, log_ri
        
    def _compute_loss(self):
        positions, target_positions, base_forces, log_tpm, log_ri = self.replay.sample()
        # shapes: pos (B,T+1,N,G), tgt (B,N,G), base_forces (B,T,N,G)
        
        gamma = float(self.args.friction)

        B, T, N, G = base_forces.shape
        v = (positions[:, 1:] - positions[:, :-1]) / self.dt          # (B,T,N,G)

        # rebuild bias per step for correct mean
        x_t = positions[:, :-1].reshape(-1, N, G)
        v_t = v.reshape(-1, N, G)  # reshape velocity to match other tensors
        tgt_t = target_positions[:, None].expand(-1, T, N, G).reshape(-1, N, G)
        
        if self.vel_conditioned:
            if self.dir_only:
                velocity_magnitude = torch.norm(v_t, dim=-1, keepdim=True)
                velocity_direction = v_t / (velocity_magnitude + 1e-8)
                B_t = self.bias_net(x_t, velocity_direction, tgt_t).view(B, T, N, G)
            else:
                B_t = self.bias_net(x_t, v_t, tgt_t).view(B, T, N, G)
        else:
            B_t = self.bias_net(x_t, tgt_t).view(B, T, N, G)

        means = (base_forces + B_t) / self.args.friction
        resid = v - means

        sigma_v = math.sqrt(2.0 * self.args.kT / (gamma * self.dt))
        sigma_v = torch.as_tensor(sigma_v, dtype=resid.dtype, device=resid.device)
        log_bpm = Normal(0.0, sigma_v).log_prob(resid).mean((1,2,3))

        # control variate
        cv = self.args.control_variate
        if cv == "global":   
            log_z = self.bias_net.log_z
        elif cv == "local":  
            log_z = (log_tpm - log_bpm).mean().detach()
        elif cv == "zero":   
            log_z = 0.0
        else: raise ValueError(cv)

        # ce objective
        if self.args.objective == "ce":
            log_rnd = (log_tpm - log_bpm.detach())
            weights = torch.softmax(log_rnd, dim=0)
            loss = -(weights * log_bpm).sum()
            
        else:  # lv loss
            loss = (log_z + log_bpm - log_tpm).pow(2).mean()

        return loss, log_ri.mean()
        
    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        
        main_batch = batch[0]["train_samples"][0]
        metric_batch = batch[0]["metric_samples"][0]
        
        x0s = main_batch["x0"][0]
        x1s = main_batch["x1_1"][0] # train on cluster 1
        N   = self.args.num_particles
        
        x0_clusters, x1_clusters, idx1 = \
            make_aligned_clusters(self.ot_sampler, x0s, x1s, N, replace=True)
        
        sample_pairs = [
            (metric_batch[0], metric_batch[1]),  # x0 → x1_1 (branch 1)
        ]
        
        batch = self._sample(x0_clusters, x1_clusters, sample_pairs)
        
        # timesteps from 0 to 1
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        loss, mean_log_ri = self._compute_loss()
        
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/mean_log_ri",
            mean_log_ri,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch[0]["val_samples"][0]
        metric_batch = batch[0]["metric_samples"][0]
        
        x0s = main_batch["x0"][0]
        x1s = main_batch["x1_2"][0] # validate on cluster 2
        N   = self.args.num_particles
        
        x0_clusters, x1_clusters, idx1 = \
            make_aligned_clusters(self.ot_sampler, x0s, x1s, N, replace=True)
        
        sample_pairs = [
            (metric_batch[0], metric_batch[1]),  # x0 -> x1_1 (branch 1)
        ]
        
        batch = self._sample(x0_clusters, x1_clusters, sample_pairs)
        
        
        # timesteps from 0 to 1
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        val_loss, mean_log_ri = self._compute_loss()
        
        
        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/mean_log_ri",
            mean_log_ri,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return val_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        

    def configure_optimizers(self):
        exclude = {id(self.bias_net.log_z)}
        params_except = [p for p in self.bias_net.parameters() if id(p) not in exclude]
        optimizer = torch.optim.AdamW(
            [
                {"params": [self.bias_net.log_z], "lr": self.args.log_z_lr},
                {"params": params_except, "lr": self.args.policy_lr},
            ]
        )

        return optimizer

class EntangledNetTrainCellUnseen(EntangledNetTrainBaseUnseen):
    @torch.no_grad()
    def plot_trajs(
        self, traj, title, fname, targets,
        timepoint_data=None, # dict with keys 't0','t1'
        cmap=None, 
        c_end='#B83CFF',
        x_label="PC1", y_label="PC2",
        save_dir=None,

    ):
                
        custom_colors_1 = ["#05009E", "#A19EFF", "#D577FF"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_cmap_1 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_1)
        custom_cmap_2 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_2)
        
        tb = traj[..., :2].detach().cpu().numpy() # (B,T+1,N,2)
        targ2 = targets[..., :2].detach().cpu().numpy() # (B,N,2)

        # optional background conversion
        def _to_np(x):
            if x is None: return None
            return x.detach().cpu().numpy() if hasattr(x, "detach") else x
        
        def to_np_2d(x):
            """Convert to np and force last dim=2 (take first two PCs). Returns None if <2 dims."""
            if x is None:
                return None
            arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            if arr.ndim == 1:  # (G,) -> (1,G)
                arr = arr[None, :]
            if arr.shape[-1] < 2:
                return None
            arr2 = arr[..., :2].reshape(-1, 2)
            return arr2

        t0_bg = to_np_2d(timepoint_data['t0']) if (timepoint_data and 't0' in timepoint_data) else None
        t1_bg = to_np_2d(timepoint_data['t1']) if (timepoint_data and 't1' in timepoint_data) else None

        # axis limits (match global-limits logic with small margins if background present)
        if t0_bg is not None or t1_bg is not None:
            coords_list = []
            if t0_bg is not None: coords_list.append(t0_bg)
            if t1_bg is not None: coords_list.append(t1_bg)
            coords_list.append(tb.reshape(-1, 2))
            coords_list.append(targ2.reshape(-1, 2))
            all_coords = np.concatenate(coords_list, axis=0)
            x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            x_margin = 0.05 * max(1e-12, (x_max - x_min))
            y_margin = 0.05 * max(1e-12, (y_max - y_min))
            x_min -= x_margin; x_max += x_margin
            y_min -= y_margin; y_max += y_margin
        else:
            x_min = y_min = -np.inf
            x_max = y_max = +np.inf

        # figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # background timepoints
        if t0_bg is not None:
            ax.scatter(t0_bg[:, 0], t0_bg[:, 1],
                    c='#05009E', s=80, alpha=0.4, marker='x',
                    label='t=0 cells', linewidth=1.5)
        if t1_bg is not None:
            ax.scatter(t1_bg[:, 0], t1_bg[:, 1],
                    c=c_end, s=80, alpha=0.4, marker='x',
                    label='t=1 cells', linewidth=1.5)

        # color map for temporal segments
        if cmap is None:
            cmap = custom_cmap_1
        num_segments = tb.shape[1]  # T+1 points -> T segments, but use T+1 for colors indexing
        colors = cmap(np.linspace(0, 1, max(2, num_segments)))

        B, T1, N, _ = tb.shape
        for b in range(B):
            for n in range(N):
                xy = tb[b, :, n, :]  # (T+1, 2)
                for t in range(T1 - 1):
                    ax.plot(
                        xy[t:t+2, 0], xy[t:t+2, 1],
                        color=colors[t], linewidth=2, alpha=0.8, zorder=2
                    )

        starts = tb[:, 0, :, :].reshape(-1, 2)     # (B*N, 2)
        ends   = tb[:, -1, :, :].reshape(-1, 2)    # (B*N, 2)
        ax.scatter(starts[:, 0], starts[:, 1],
                c='#05009E', s=30, marker='o', label='Trajectory Start',
                zorder=5, edgecolors='white', linewidth=1)
        ax.scatter(ends[:, 0], ends[:, 1],
                c=c_end, s=30, marker='o', label='Trajectory End',
                zorder=5, edgecolors='white', linewidth=1)
        
        tars = targ2.reshape(-1, 2)
        ax.scatter(tars[:, 0], tars[:, 1],
                s=24, marker='x', linewidths=1.5, c='#B83CFF',
                alpha=0.7, label='targets', zorder=4)

        if np.isfinite(x_min):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if save_dir is None:
            out_path = os.path.join(self.args.save_dir, "figures", self.args.data_name)
            os.makedirs(out_path, exist_ok=True)
        else:
            out_path = save_dir
            os.makedirs(out_path, exist_ok=True)

        fpath = os.path.join(out_path, fname)
        plt.savefig(fpath, dpi=300)
        print(f"figure saved: {fpath}")
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        main_batch = batch[0]["test_samples"][0]
        x0s = main_batch["x0"][0] 
        x1_1s = main_batch["x1_1"][0] # training endpoint
        x1_2s = main_batch["x1_2"][0] # unseen endpoint
        #x1s = torch.cat([x1_1s, x1_2s], dim=0) # (M1+M2,G) or (B,G)
        dataset_full = main_batch["dataset"][0] # (M,G)

        device = x0s.device

        #B, N, G = x0s.shape
        N    = self.args.num_particles
        T    = self.args.num_steps
        G    = self.args.dim
        gamma = float(self.args.friction)
        kT    = float(getattr(self.args, "kT", 0.0))
        ell   = float(getattr(self.args, "adj_length_scale", 1.0))
        noise_scale = (0.0 if kT <= 0 else math.sqrt(2.0 * kT * self.dt / gamma))

        # Ensure pools are (M,G)
        if x1_1s.ndim == 3: x1_1s = x1_1s.reshape(-1, G)
        if x1_2s.ndim == 3: x1_2s = x1_2s.reshape(-1, G)
        
        x0c_1, x1c_1, _ = make_aligned_clusters(self.ot_sampler, x0s, x1_1s, N, replace=True)   # to training endpoint
        x0c_2, x1c_2, _ = make_aligned_clusters(self.ot_sampler, x0s, x1_2s, N, replace=True)   # to unseen endpoint

        B = x0c_1.shape[0]

        # rollout base-only dynamics (no bias)
        def rollout_base(x0c, x1c):
            pos  = x0c.clone()
            traj = torch.empty((B, T+1, N, G), dtype=pos.dtype, device=pos.device)
            traj[:, 0] = pos.clone()
            for t in range(T):

                # manifold/natural gradient towards target
                F_nat, _ = natural_gradient_force(
                    self.data_manifold_metric, pos, metric_samples=None, timestep=t
                )                                             # (B,N,G)
                base_force = F_nat
                xi = torch.randn_like(pos) * noise_scale if noise_scale > 0 else 0.0
                pos = pos + (self.dt / gamma) * base_force + xi
                traj[:, t+1] = pos.clone()
            return traj  # (B,T+1,N,G)

        # rollout bias-only
        def rollout_bias_only(x0c, x1c):
            pos  = x0c.clone()
            traj = torch.empty((B, T+1, N, G), dtype=pos.dtype, device=pos.device)
            traj[:, 0] = pos.clone()
            for t in range(T):
                if t == 0: 
                    velocity = torch.zeros_like(pos)
                else:
                    velocity = (pos - traj[:, t-1]) / self.dt
                
                if self.vel_conditioned:
                    # Use velocity direction (unit vector) instead of full velocity
                    if self.dir_only:
                        velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
                        velocity_direction = velocity / (velocity_magnitude + 1e-8)
                        bias_force = self.bias_net(
                            pos.detach(), velocity_direction.detach(), x1c.detach()
                        ).detach()    
                    else:
                        bias_force = self.bias_net(
                            pos.detach(), velocity.detach(), x1c.detach()
                        ).detach()
                else:
                    bias_force = self.bias_net(
                        pos.detach(), x1c.detach()
                    ).detach()   # (B,N,G)
                    
                xi = torch.randn_like(pos) * noise_scale if noise_scale > 0 else 0.0
                pos = pos + (self.dt / gamma) * (bias_force) + xi
                traj[:, t+1] = pos.clone()
            return traj  # (B,T+1,N,G)

        # rollout bias+base (controlled dynamics)
        def rollout_bias_plus_base(x0c, x1c):
            pos  = x0c.clone()
            traj = torch.empty((B, T+1, N, G), dtype=pos.dtype, device=pos.device)
            traj[:, 0] = pos.clone()
            for t in range(T):
                if t == 0: 
                    velocity = torch.zeros_like(pos)
                else:
                    velocity = (pos - traj[:, t-1]) / self.dt
                    
                F_nat, _ = natural_gradient_force(
                    self.data_manifold_metric, pos, metric_samples=None, timestep=t
                ) # (B,N,G)
                base_force = F_nat
                
                if self.vel_conditioned:
                    if self.dir_only:
                        velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
                        velocity_direction = velocity / (velocity_magnitude + 1e-8)
                        bias_force = self.bias_net(
                            pos.detach(), velocity_direction.detach(), x1c.detach()
                        ).detach()    
                    else:
                        bias_force = self.bias_net(
                            pos.detach(), velocity.detach(), x1c.detach()
                        ).detach()
                else:
                    bias_force = self.bias_net(
                        pos.detach(), x1c.detach()
                    ).detach()   # (B,N,G)
                    
                xi = torch.randn_like(pos) * noise_scale if noise_scale > 0 else 0.0
                pos = pos + (self.dt / gamma) * (base_force + bias_force) + xi
                traj[:, t+1] = pos.clone()
            return traj  # (B,T+1,N,G)


        traj_to_x1_1 = rollout_base(x0c_1, x1c_1)
        traj_to_x1_2 = rollout_base(x0c_2, x1c_2)

        # New:
        traj_bias_only_x1_1 = rollout_bias_only(x0c_1, x1c_1)
        traj_bias_plus_x1_1 = rollout_bias_plus_base(x0c_1, x1c_1)
        traj_bias_only_x1_2 = rollout_bias_only(x0c_2, x1c_2)
        traj_bias_plus_x1_2 = rollout_bias_plus_base(x0c_2, x1c_2)


        # ---------- 4) Plot (first two dims) with targets as blue X ----------
        save_dir = os.path.join(self.args.save_dir, "figures", self.args.data_name)
        os.makedirs(save_dir, exist_ok=True)

        ds2 = (dataset_full[:, :2].detach().cpu().numpy()
               if isinstance(dataset_full, torch.Tensor) else dataset_full[:, :2])
            
        custom_colors_1 = ["#05009E", "#A19EFF", "#6B67EE"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_cmap_1 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_1)
        custom_cmap_2 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_2)
        
        t0_data = self.timepoint_data["t0"]
        t1_data = torch.cat([self.timepoint_data["t1_1"], self.timepoint_data["t1_2"]], 
                            dim=0)
        
        # seen endpoint
        self.plot_trajs(
            traj_bias_only_x1_1,
            "Bias-only → training endpoint (x1_1)",
            f"{self.args.data_name}_bias_only_to_x1_1.png",
            x1c_1,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_1,
            c_end='#6B67EE',
            save_dir=save_dir
        )

        self.plot_trajs(traj_to_x1_1,
            "Base-only trajectories → training endpoint (x1_1)",
            f"{self.args.data_name}_base_only_to_x1_1.png",
            x1c_1,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_1,
            c_end='#6B67EE',
            save_dir=save_dir
        )
        
        self.plot_trajs(traj_bias_plus_x1_1,
            "Bias + base trajectories → training endpoint (x1_1)",
            f"{self.args.data_name}_bias_plus_base_to_x1_1.png",
            x1c_1,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_1,
            c_end='#6B67EE',
            save_dir=save_dir
        )

        # unseen endpoint
        self.plot_trajs(traj_to_x1_2,
            "Base-only trajectories → unseen endpoint (x1_2)",
            f"{self.args.data_name}_base_only_to_x1_2.png",
            x1c_2,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_2,
            c_end='#50B2D7',
            save_dir=save_dir
        )

        self.plot_trajs(traj_bias_only_x1_2,
            "Bias-only trajectories → unseen endpoint (x1_2)",
            f"{self.args.data_name}_bias_only_to_x1_2.png",
            x1c_2,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_2,
            c_end='#50B2D7',
            save_dir=save_dir
        )

        self.plot_trajs(traj_bias_plus_x1_2,
            "Bias + base trajectories → unseen endpoint (x1_2)",
            f"{self.args.data_name}_bias_plus_base_to_x1_2.png",
            x1c_2,
            timepoint_data={"t0": t0_data, "t1": t1_data},
            cmap=custom_cmap_2,
            c_end='#50B2D7',
            save_dir=save_dir
        )

        def eval_cluster_set(traj_B_T1_N_G, targets_B_N_G, tag: str):
            
            finals = traj_B_T1_N_G[:, -1] # (B, N, G)
            B = finals.shape[0]

            # accumulate per-anchor metrics, then mean over B
            per_b_vals = []
            for b in range(B):
                pred_b = finals[b].to(device) # (N, G)
                true_b = targets_B_N_G[b].to(device)  # (N, G)
                
                names, vals = compute_distribution_distances(pred_b, true_b)
                per_b_vals.append(vals)

            per_b_vals = np.asarray(per_b_vals, dtype=float) # (B, num_metrics)
            mean_vals   = per_b_vals.mean(axis=0)
            std_vals    = per_b_vals.std(axis=0, ddof=1) if B > 1 else np.zeros_like(mean_vals)

            # log (Lightning-friendly); also return a dict
            out = {}
            for n, m, s in zip(names, mean_vals, std_vals):
                self.log(f"test/{tag}/{n}",  float(m), on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f"test/{tag}/{n}_std", float(s), on_step=False, on_epoch=True, prog_bar=False, logger=True)
                out[f"{n}"] = float(m)
                out[f"{n}_std"] = float(s)
            return out


        def eval_vs_full_distribution(traj_B_T1_N_G, full_target_distribution, tag: str):
            
            finals = traj_B_T1_N_G[:, -1] # (B, N, G)
            
            # Flatten all trajectory endpoints into single distribution
            pred_all = finals.reshape(-1, G).to(device)  # (B*N, G)
            true_all = full_target_distribution.to(device)  # (M, G)
            
            # Ensure both tensors are on the same device and have the right dtype
            pred_all = pred_all.float()
            true_all = true_all.float()
            
            # Resample to same size for MMD compatibility
            names, vals = compute_wasserstein_distances(pred_all, true_all)
            
            # log results
            out = {}
            for n, v in zip(names, vals):
                self.log(f"test/{tag}/{n}", float(v), on_step=False, on_epoch=True, prog_bar=False, logger=True)
                out[f"{n}"] = float(v)
            return out

        # quantitative metrics over multiple independent simulations
        num_sims = 5
        base_seed = 12345

        def run_one_sim(sim_idx: int):
            # re-rollout all six variants with a distinct RNG state
            devices = [device.index] if device.type == "cuda" and device.index is not None else None
            with torch.random.fork_rng(devices=devices):
                
                torch.manual_seed(base_seed + sim_idx)

                tr_base_11 = rollout_base(x0c_1, x1c_1)
                tr_base_12 = rollout_base(x0c_2, x1c_2)

                tr_bias_only_11 = rollout_bias_only(x0c_1, x1c_1)
                tr_bias_plus_11 = rollout_bias_plus_base(x0c_1, x1c_1)

                tr_bias_only_12 = rollout_bias_only(x0c_2, x1c_2)
                tr_bias_plus_12 = rollout_bias_plus_base(x0c_2, x1c_2)

            # evaluate per-cluster metrics (your helper returns a dict of means/std over B)
            metrics = {}
            metrics.update({f"base_x1_1/{k}": v for k, v in eval_cluster_set(tr_base_11,      x1c_1, "base/x1_1").items()})
            metrics.update({f"base_x1_2/{k}": v for k, v in eval_cluster_set(tr_base_12,      x1c_2, "base/x1_2").items()})
            metrics.update({f"bias_only_x1_1/{k}": v for k, v in eval_cluster_set(tr_bias_only_11, x1c_1, "bias_only/x1_1").items()})
            metrics.update({f"bias_plus_x1_1/{k}": v for k, v in eval_cluster_set(tr_bias_plus_11, x1c_1, "bias_plus/x1_1").items()})
            metrics.update({f"bias_only_x1_2/{k}": v for k, v in eval_cluster_set(tr_bias_only_12, x1c_2, "bias_only/x1_2").items()})
            metrics.update({f"bias_plus_x1_2/{k}": v for k, v in eval_cluster_set(tr_bias_plus_12, x1c_2, "bias_plus/x1_2").items()})
            
            # evaluate vs full distributions
            metrics.update({f"base_vs_full_x1_1/{k}": v for k, v in eval_vs_full_distribution(tr_base_11, x1_1s, "base_vs_full/x1_1").items()})
            metrics.update({f"base_vs_full_x1_2/{k}": v for k, v in eval_vs_full_distribution(tr_base_12, x1_2s, "base_vs_full/x1_2").items()})
            metrics.update({f"bias_only_vs_full_x1_1/{k}": v for k, v in eval_vs_full_distribution(tr_bias_only_11, x1_1s, "bias_only_vs_full/x1_1").items()})
            metrics.update({f"bias_plus_vs_full_x1_1/{k}": v for k, v in eval_vs_full_distribution(tr_bias_plus_11, x1_1s, "bias_plus_vs_full/x1_1").items()})
            metrics.update({f"bias_only_vs_full_x1_2/{k}": v for k, v in eval_vs_full_distribution(tr_bias_only_12, x1_2s, "bias_only_vs_full/x1_2").items()})
            metrics.update({f"bias_plus_vs_full_x1_2/{k}": v for k, v in eval_vs_full_distribution(tr_bias_plus_12, x1_2s, "bias_plus_vs_full/x1_2").items()})
            
            return metrics

        # run K sims and collect metrics per run
        metrics_runs = [run_one_sim(k) for k in range(num_sims)]

        agg = {}
        all_keys = sorted(metrics_runs[0].keys())

        for key in all_keys:
            vals = torch.tensor([m[key] for m in metrics_runs], dtype=torch.float32)
            agg[f"{key}_runs_mean"] = vals.mean().item()
            agg[f"{key}_runs_std"]  = (vals.std(unbiased=True).item() if num_sims > 1 else 0.0)

            # Log for Lightning
            self.log(f"test/{key}_runs_mean", agg[f"{key}_runs_mean"], on_step=False, on_epoch=True, logger=True)
            self.log(f"test/{key}_runs_std",  agg[f"{key}_runs_std"],  on_step=False, on_epoch=True, logger=True)

        if getattr(self.trainer, "is_global_zero", True):
            out_dir = os.path.join(self.args.save_dir, "metrics", self.args.data_name)
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            with open(os.path.join(out_dir, f"test_metrics_aggregated_{num_sims}runs_{stamp}.json"), "w") as f:
                json.dump(agg, f, indent=2)
        
        return {"num_traj": B}

### REPLAY BUFFER ###     
class ReplayBuffer:
    def __init__(self, args):
        BZ, T, N, G = args.buffer_size, args.num_steps, args.num_particles, args.dim
        self.positions = torch.zeros((BZ, T+1, N, G), device=args.device)
        self.target_positions = torch.zeros((BZ, N,   G), device=args.device)
        self.forces = torch.zeros((BZ, T,   N, G), device=args.device)
        self.log_tpm = torch.zeros((BZ,), device=args.device)
        self.rewards = torch.zeros((BZ,), device=args.device)
        self.batch_size, self.buffer_size, self.count = args.batch_size, args.buffer_size, 0

    def add_ranked(self, data):
        positions, target_positions, forces, log_tpm, rewards = data
        # concat existing + new, keep top by reward
        if self.count > 0:
            pos = torch.cat([self.positions[:self.count], positions], dim=0)
            tgt = torch.cat([self.target_positions[:self.count], target_positions], dim=0)
            frc = torch.cat([self.forces[:self.count], forces], dim=0)
            ltp = torch.cat([self.log_tpm[:self.count], log_tpm], dim=0)
            rwd = torch.cat([self.rewards[:self.count], rewards], dim=0)
        else:
            pos, tgt, frc, ltp, rwd = positions, target_positions, forces, log_tpm, rewards
        k = min(self.buffer_size, pos.size(0))
        top_vals, top_idx = torch.topk(rwd, k=k, largest=True, sorted=False)
        self.positions[:k] = pos.index_select(0, top_idx)
        self.target_positions[:k] = tgt.index_select(0, top_idx)
        self.forces[:k] = frc.index_select(0, top_idx)
        self.log_tpm[:k] = ltp.index_select(0, top_idx)
        self.rewards[:k] = top_vals
        self.count = k

    def sample(self):
        assert self.count > 0, "buffer is empty"
        idx = torch.randint(0, self.count, (self.batch_size,), device=self.positions.device)
        return (
            self.positions[idx], self.target_positions[idx],
            self.forces[idx], self.log_tpm[idx], self.rewards[idx],
        )

### PATH OBJECTIVE ###
class PathObjective:
    def __init__(self, args):
        self.dt = float(1.0 / args.num_steps)
        self.gamma = args.friction
        self.kT = getattr(args, "kT", 0.0)
        self.sigma_v = math.sqrt(2.0 * self.kT / (self.gamma * self.dt))
        self.log_prob = Normal(0.0, self.sigma_v).log_prob
        self.sigma = args.sigma

    def __call__(self, positions, target_positions, base_forces):
        log_upm = self.unbiased_path_measure(positions, base_forces)
        log_ri, final_idx = self.relaxed_indicator(positions, target_positions)
        return log_upm + log_ri, final_idx, log_ri

    def unbiased_path_measure(self, positions, base_forces):
        v = (positions[:, 1:] - positions[:, :-1]) / self.dt # (B,T,N,G)
        
        means = base_forces / self.gamma # (B,T,N,G)
        resid = v - means
        return self.log_prob(resid).mean((1,2,3))

    def relaxed_indicator(self, positions, target_positions):
        # allow (B,N,G) or (B,T+1,N,G)
        device = positions.device
        if target_positions.ndim == 3:
            target_positions = target_positions[:, None].expand_as(positions)
            
        target_positions = target_positions.to(device)
        dist2  = (positions - target_positions).pow(2).mean((-2,-1)) # (B,T+1)
        log_ri = -0.5 * dist2 / (self.sigma**2)
        vals, idx = log_ri.max(dim=1)
        return vals, idx