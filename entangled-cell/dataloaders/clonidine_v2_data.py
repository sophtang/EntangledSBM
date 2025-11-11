import torch
import sys
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import pandas as pd
import numpy as np
from functools import partial
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset

class ClonidineV2DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = args.batch_size
        self.max_dim = args.dim
        #self.whiten = args.whiten
        self.split_ratios = args.split_ratios
        
        self.dim = args.dim
        # Path to your combined data
        self.data_path = f"{args.root_dir}/data/pca_and_leiden_labels.csv"
        self.num_timesteps = 2
        self.args = args
        self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.data_path, comment='#')
        df = df.iloc[:, 1:]
        df = df.replace('', np.nan)
        pc_cols = df.columns[:self.dim]
        for col in pc_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        leiden_dmso_col = 'leiden_DMSO_TF_0.0uM'
        leiden_clonidine_col = 'leiden_Clonidine (hydrochloride)_5.0uM'

        dmso_mask = df[leiden_dmso_col].notna()        # Has leiden value in DMSO column
        clonidine_mask = df[leiden_clonidine_col].notna()  # Has leiden value in Clonidine column
        
        dmso_data = df[dmso_mask].copy()
        clonidine_data = df[clonidine_mask].copy()
        
        top_clonidine_clusters = ['0.0', '4.0']
        
        x1_1_data = clonidine_data[clonidine_data[leiden_clonidine_col].astype(str) == top_clonidine_clusters[0]]
        x1_2_data = clonidine_data[clonidine_data[leiden_clonidine_col].astype(str) == top_clonidine_clusters[1]]
        
        x1_1_coords = x1_1_data[pc_cols].values 
        x1_2_coords = x1_2_data[pc_cols].values 
        
        x1_1_coords = x1_1_coords.astype(float)
        x1_2_coords = x1_2_coords.astype(float)

        # Target size is now the minimum across all three endpoint clusters
        target_size = min(len(x1_1_coords), len(x1_2_coords),)
        
        # Helper function to select points closest to centroid
        def select_closest_to_centroid(coords, target_size):
            if len(coords) <= target_size:
                return coords
            
            # Calculate centroid
            centroid = np.mean(coords, axis=0)
            
            # Calculate distances to centroid
            distances = np.linalg.norm(coords - centroid, axis=1)
            
            # Get indices of closest points
            closest_indices = np.argsort(distances)[:target_size]
            
            return coords[closest_indices]
        
        # Sample all endpoint clusters to target size using centroid-based selection
        x1_1_coords = select_closest_to_centroid(x1_1_coords, target_size)
        x1_2_coords = select_closest_to_centroid(x1_2_coords, target_size)
        
        dmso_cluster_counts = dmso_data[leiden_dmso_col].value_counts()
        
        # DMSO (unchanged)
        largest_dmso_cluster = dmso_cluster_counts.index[0]
        dmso_cluster_data = dmso_data[dmso_data[leiden_dmso_col] == largest_dmso_cluster]
        
        dmso_coords = dmso_cluster_data[pc_cols].values
        
        # Random sampling from largest DMSO cluster to match target size
        # For DMSO, we'll also use centroid-based selection for consistency
        if len(dmso_coords) >= target_size:
            x0_coords = select_closest_to_centroid(dmso_coords, target_size)
        else:
            # If largest cluster is smaller than target, use all of it and pad with other DMSO cells
            remaining_needed = target_size - len(dmso_coords)
            other_dmso_data = dmso_data[dmso_data[leiden_dmso_col] != largest_dmso_cluster]
            other_dmso_coords = other_dmso_data[pc_cols].values
            
            if len(other_dmso_coords) >= remaining_needed:
                # Select closest to centroid from other DMSO cells
                other_selected = select_closest_to_centroid(other_dmso_coords, remaining_needed)
                x0_coords = np.vstack([dmso_coords, other_selected])
            else:
                # Use all available DMSO cells and reduce target size
                all_dmso_coords = dmso_data[pc_cols].values
                target_size = min(target_size, len(all_dmso_coords))
                x0_coords = select_closest_to_centroid(all_dmso_coords, target_size)
                
                # Re-select endpoint clusters with updated target size
                x1_1_coords = select_closest_to_centroid(x1_1_data[pc_cols].values.astype(float), target_size)
                x1_2_coords = select_closest_to_centroid(x1_2_data[pc_cols].values.astype(float), target_size)
        
        # No need to resample since we already selected the right number
        # The endpoint clusters are already at target_size from centroid-based selection
        
        self.n_samples = target_size
        
        x0 = torch.tensor(x0_coords, dtype=torch.float32)
        x1_1 = torch.tensor(x1_1_coords, dtype=torch.float32)
        x1_2 = torch.tensor(x1_2_coords, dtype=torch.float32)
        
        self.coords_t0 = x0
        self.coords_t1_1 = x1_1
        self.coords_t1_2 = x1_2
        self.time_labels = np.concatenate([
            np.zeros(len(self.coords_t0)),    # t=0
            np.ones(len(self.coords_t1_1)),     # t=1
            np.ones(len(self.coords_t1_2)),
        ])
        
        split_index = int(target_size * self.split_ratios[0])
        
        if target_size - split_index < self.batch_size:
            split_index = target_size - self.batch_size
        
        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        train_x1_1 = x1_1[:split_index]
        val_x1_1 = x1_1[split_index:]
        train_x1_2 = x1_2[:split_index]
        val_x1_2 = x1_2[split_index:]

        
        self.val_x0 = val_x0
        self.val_x1_1 = val_x1_1
        self.val_x1_2 = val_x1_2
        
        train_x0_weights = torch.full((train_x0.shape[0], 1), fill_value=1.0)
        train_x1_1_weights = torch.full((train_x1_1.shape[0], 1), fill_value=0.5)
        train_x1_2_weights = torch.full((train_x1_2.shape[0], 1), fill_value=0.5)
        
        val_x0_weights = torch.full((val_x0.shape[0], 1), fill_value=1.0)
        val_x1_1_weights = torch.full((val_x1_1.shape[0], 1), fill_value=0.5)
        val_x1_2_weights = torch.full((val_x1_2.shape[0], 1), fill_value=0.5)
        
        # Updated train dataloaders to include x1_3
        self.train_dataloaders = {
            "x0": DataLoader(TensorDataset(train_x0, train_x0_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_1": DataLoader(TensorDataset(train_x1_1, train_x1_1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(train_x1_2, train_x1_2_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        
        self.val_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.batch_size, shuffle=False, drop_last=True),
            "x1_1": DataLoader(TensorDataset(val_x1_1, val_x1_1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(val_x1_2, val_x1_2_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        
        all_coords = df[pc_cols].dropna().values.astype(float)
        self.dataset = torch.tensor(all_coords, dtype=torch.float32)
        self.tree = cKDTree(all_coords)

        self.test_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.val_x0.shape[0], shuffle=False, drop_last=False),
            "x1_1": DataLoader(TensorDataset(val_x1_1, val_x1_1_weights), batch_size=self.val_x1_1.shape[0], shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(val_x1_2, val_x1_2_weights), batch_size=self.val_x1_2.shape[0], shuffle=True, drop_last=True),
            "dataset": DataLoader(TensorDataset(self.dataset), batch_size=self.dataset.shape[0], shuffle=False, drop_last=False),
        }

        # Updated metric samples - now using 4 clusters instead of 3
        #km_all = KMeans(n_clusters=4, random_state=42).fit(self.dataset.numpy())
        """km_all = KMeans(n_clusters=3, random_state=0).fit(self.dataset.numpy())

        cluster_labels = km_all.labels_
        
        cluster_0_mask = cluster_labels == 0
        cluster_1_mask = cluster_labels == 1
        cluster_2_mask = cluster_labels == 2
        
        samples = self.dataset.cpu().numpy()
            
        cluster_0_data = samples[cluster_0_mask]
        cluster_1_data = samples[cluster_1_mask]
        cluster_2_data = samples[cluster_2_mask]
                
        self.metric_samples_dataloaders = [
            DataLoader(
                torch.tensor(cluster_2_data, dtype=torch.float32),
                batch_size=cluster_2_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                torch.tensor(cluster_0_data, dtype=torch.float32),
                batch_size=cluster_0_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            
            DataLoader(
                torch.tensor(cluster_1_data, dtype=torch.float32),
                batch_size=cluster_1_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]"""
        
        # Updated metric samples - now using 4 clusters instead of 3
        #km_all = KMeans(n_clusters=4, random_state=42).fit(self.dataset.numpy())
        km_all = KMeans(n_clusters=2, random_state=0).fit(self.dataset.numpy())

        cluster_labels = km_all.labels_
        
        cluster_0_mask = cluster_labels == 0
        cluster_1_mask = cluster_labels == 1
        
        samples = self.dataset.cpu().numpy()
            
        cluster_0_data = samples[cluster_0_mask]
        cluster_1_data = samples[cluster_1_mask]
                
        self.metric_samples_dataloaders = [
            DataLoader(
                torch.tensor(cluster_1_data, dtype=torch.float32),
                batch_size=cluster_1_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                torch.tensor(cluster_0_data, dtype=torch.float32),
                batch_size=cluster_0_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]
        
    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    
    
    def test_dataloader(self):
        combined_loaders = {
            "test_samples": CombinedLoader(self.test_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def get_manifold_proj(self, points):
        """Adapted for 2D cell data - uses local neighborhood averaging instead of plane fitting"""
        return partial(self.local_smoothing_op, tree=self.tree, dataset=self.dataset)

    @staticmethod
    def local_smoothing_op(x, tree, dataset, k=10, temp=1e-3):
        """
        Apply local smoothing based on k-nearest neighbors in the full dataset
        This replaces the plane projection for 2D manifold regularization
        """
        points_np = x.detach().cpu().numpy()
        _, idx = tree.query(points_np, k=k)
        nearest_pts = dataset[idx]  # Shape: (batch_size, k, 2)
        
        # Compute weighted average of neighbors
        dists = (x.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted average of neighbors
        smoothed = (weights * nearest_pts).sum(dim=1)
        
        # Blend original point with smoothed version
        alpha = 0.3  # How much smoothing to apply
        return (1 - alpha) * x + alpha * smoothed
    
    def get_timepoint_data(self):
        """Return data organized by timepoints for visualization"""
        return {
            't0': self.coords_t0,
            't1_1': self.coords_t1_1, 
            't1_2': self.coords_t1_2, 
            'time_labels': self.time_labels
        }
    
def get_datamodule():
    datamodule = ClonidineV2DataModule(args)
    datamodule.setup(stage="fit")   
    return datamodule