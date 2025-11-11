### Adapted from Metric Flow Matching (https://github.com/kkapusniak/metric-flow-matching)

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader


from geo_metrics.land import land_metric_tensor
from geo_metrics.rbf_unseen import RBFNetwork

def unconditional_landscape_force(
    data_manifold_metric, position, metric_samples, timestep=0, 
    landscape_mode="metric_potential", eps=1e-6, unit_speed=False, target_speed=None
):
    # Call natural_gradient_force with dummy grad_phi
    dummy_grad_phi = torch.zeros_like(position)
    return natural_gradient_force(
        data_manifold_metric, position, dummy_grad_phi, metric_samples, 
        timestep=timestep, eps=eps, unit_speed=unit_speed, 
        target_speed=target_speed, landscape_mode=landscape_mode
    )

def manifold_metric_diag(data_manifold_metric, position, metric_samples, timestep):
    
    B, N, G = position.shape
    x_flat = position.reshape(B*N, G)  # (B*N, G)
    M_flat = data_manifold_metric.calculate_metric(x_flat, metric_samples, timestep)  # (B*N, G)
    return M_flat.view(B, N, -1)  # reshape back


def natural_gradient_force(
    data_manifold_metric, position, metric_samples, timestep=0,
    eps=1e-6, unit_speed=False, target_speed=None
):
    B, N, G = position.shape
    M_dd = manifold_metric_diag(data_manifold_metric, position, metric_samples, timestep=0)  # (B,N,G)
    
    position_grad = position.clone().requires_grad_(True)
    M_dd_grad = manifold_metric_diag(data_manifold_metric, position_grad, metric_samples, timestep=0)
    
    # Create potential from metric: U = -sum(log(M_ii + eps))
    potential = -(M_dd_grad + eps).log().sum()
    
    try:
        grad_potential = torch.autograd.grad(
            outputs=potential, 
            inputs=position_grad, 
            create_graph=False, 
            retain_graph=False,
            allow_unused=True
        )[0]
        
        if grad_potential is not None:
            u = -grad_potential / (M_dd.detach() + eps)
        else:
            u = torch.randn_like(position) / (M_dd + eps)
    except:
        u = torch.randn_like(position) / (M_dd + eps)

    speed_M = torch.sqrt((u.pow(2) * M_dd.detach()).sum(dim=-1, keepdim=True).clamp_min(1e-24))  # (B,N,1)

    if unit_speed:
        if target_speed is None:
            u = u / speed_M.clamp_min(1e-12)
        else:
            ts = (target_speed if torch.is_tensor(target_speed)
                  else torch.tensor(target_speed, dtype=u.dtype, device=u.device))
            u = u * (ts.view(1, 1, 1) / speed_M.clamp_min(1e-12))

    return u, speed_M


class DataManifoldMetric:
    def __init__(
        self,
        args,
        skipped_time_points=None,
        datamodule=None,
    ):
        self.skipped_time_points = skipped_time_points
        self.datamodule = datamodule

        self.gamma = args.gamma
        self.rho = args.rho
        self.metric = args.velocity_metric
        self.n_centers = args.n_centers
        self.kappa = args.kappa
        self.metric_epochs = args.metric_epochs
        self.metric_patience = args.metric_patience
        self.lr = args.metric_lr
        self.alpha_metric = args.alpha_metric
        self.image_data = args.data_type == "image"
        self.accelerator = args.accelerator

        self.called_first_time = True
        self.args = args

    def calculate_metric(self, x_t, samples, current_timestep):
        if self.metric == "land":
            M_dd_x_t = (
                land_metric_tensor(x_t, samples, self.gamma, self.rho)
                ** self.alpha_metric
            )
            
        elif self.metric == "rbf":
            if self.called_first_time:
                self.rbf_networks = []
                for timestep in range(self.datamodule.num_timesteps - 1):
                    if timestep in self.skipped_time_points:
                        continue
                    print("Learning RBF networks, timestep: ", timestep)
                    rbf_network = RBFNetwork(
                        current_timestep=timestep,
                        next_timestep=timestep
                        + 1
                        + (1 if timestep + 1 in self.skipped_time_points else 0),
                        n_centers=self.n_centers,
                        kappa=self.kappa,
                        lr=self.lr,
                        datamodule=self.datamodule,
                        args=self.args
                    )
                    early_stop_callback = pl.callbacks.EarlyStopping(
                        monitor="MetricModel/val_loss_learn_metric",
                        patience=self.metric_patience,
                        mode="min",
                    )
                    trainer = pl.Trainer(
                        max_epochs=self.metric_epochs,
                        accelerator=self.accelerator,
                        logger=WandbLogger(),
                        num_sanity_val_steps=0,
                        callbacks=(
                            [early_stop_callback] if not self.image_data else None
                        ),
                    )
                    
                    trainer.fit(rbf_network, self.datamodule)
                    
                    self.rbf_networks.append(rbf_network)
                self.called_first_time = False
                print("Learning RBF networksss... Done")
            M_dd_x_t = self.rbf_networks[current_timestep].compute_metric(
                x_t,
                epsilon=self.rho,
                alpha=self.alpha_metric,
                image_hx=self.image_data,
            )
        return M_dd_x_t

    def calculate_velocity(self, x_t, u_t, samples, timestep):

        if len(u_t.shape) > 2:
            u_t = u_t.reshape(u_t.shape[0], -1)
            x_t = x_t.reshape(x_t.shape[0], -1)
        M_dd_x_t = self.calculate_metric(x_t, samples, timestep).to(u_t.device)

        velocity = torch.sqrt(((u_t**2) * M_dd_x_t).sum(dim=-1))
        ut_sum = (u_t**2).sum(dim=-1)
        metric_sum = M_dd_x_t.sum(dim=-1)
        return velocity, ut_sum, metric_sum
