import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
import numpy as np


class RBFNetwork(pl.LightningModule):
    def __init__(
        self,
        current_timestep,
        next_timestep,
        n_centers: int = 100,
        kappa: float = 1.0,
        lr=1e-2,
        datamodule=None,
        image_data=False,
        args=None
    ):
        super().__init__()
        self.K = n_centers
        self.current_timestep = current_timestep
        self.next_timestep = next_timestep
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.last_val_loss = 1
        self.lr = lr
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.datamodule = datamodule
        self.image_data = image_data
        self.args = args

    def on_before_zero_grad(self, *args, **kwargs):
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    def on_train_start(self):
        with torch.no_grad():
            
            batch = next(iter(self.trainer.datamodule.train_dataloader()))
            """metric_samples_batch_filtered = [
                x
                for i, x in enumerate(batch[0]["metric_samples"][0])
                if i in [self.current_timestep, self.next_timestep]
            ]"""
            metric_samples = batch[0]["metric_samples"][0]
            all_data = torch.cat(metric_samples)
            data_to_fit = all_data

            print("Fitting Clustering model...")
            self.clustering_model.fit(data_to_fit)

            clusters = (
                self.calculate_centroids(all_data, self.clustering_model.labels_)
                if self.image_data
                else self.clustering_model.cluster_centers_
            )

            self.C = torch.tensor(clusters, dtype=torch.float32).to(self.device)
            labels = self.clustering_model.labels_
            sigmas = np.zeros((self.K, 1))

            for k in range(self.K):
                points = all_data[labels == k, :]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                sigmas[k, :] = np.sqrt(
                    variance.sum() if self.image_data else variance.mean()
                )

            self.lamda = torch.tensor(
                0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32
            ).to(self.device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1).to(self.C.device)
        
        x = x.to(self.C.device)
        dist2 = torch.cdist(x, self.C) ** 2
        self.phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        
        h_x = (self.W.to(x.device) * self.phi_x).sum(dim=1)
        return h_x

    def training_step(self, batch, batch_idx):
        main_batch = batch[0]["train_samples"][0]
        
        x0 = main_batch["x0"][0]
        if not self.args.unseen:
            x1 = main_batch["x1"][0]
            inputs = torch.cat([x0, x1], dim=0).to(self.device)
        else:
            x1_1 = main_batch["x1_1"][0]
            x1_2 = main_batch["x1_2"][0]
            
            inputs = torch.cat([x0, x1_1, x1_2], dim=0).to(self.device)
        
        loss = ((1 - self.forward(inputs)) ** 2).mean()
        self.log(
            "MetricModel/train_loss_learn_metric",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch[0]["val_samples"][0]
        
        x0 = main_batch["x0"][0]
        if not self.args.unseen:
            x1 = main_batch["x1"][0]
            inputs = torch.cat([x0, x1], dim=0).to(self.device)
        else:
            x1_1 = main_batch["x1_1"][0]
            x1_2 = main_batch["x1_2"][0]
            
            inputs = torch.cat([x0, x1_1, x1_2], dim=0).to(self.device)

        h = self.forward(inputs)
        
        loss = ((1 - h) ** 2).mean()
        self.log(
            "MetricModel/val_loss_learn_metric",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.last_val_loss = loss.detach()
        return loss

    def calculate_centroids(self, all_data, labels):
        unique_labels = np.unique(labels)
        centroids = np.zeros((len(unique_labels), all_data.shape[1]))
        for i, label in enumerate(unique_labels):
            centroids[i] = all_data[labels == label].mean(axis=0)
        return centroids

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_metric(self, x, alpha=1, epsilon=1e-2, image_hx=False):
        if epsilon < 0:
            epsilon = (1 - self.last_val_loss.item()) / abs(epsilon)
        h_x = self.forward(x)
        if image_hx:
            h_x = 1 - torch.abs(1 - h_x)
            M_x = 1 / (h_x**alpha + epsilon)
        else:
            M_x = 1 / (h_x + epsilon) ** alpha
        return M_x
