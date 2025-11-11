import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasForceTransformer(nn.Module):
    def __init__(self,
                 args,
                 d_model = 256,
                 nhead = 8,
                 num_layers = 4,
                 dim_feedforward = 512,
                 dropout = 0.1,
                 ):
        super().__init__()
        self.device = args.device
        self.N = args.num_particles
        
        self.use_delta_to_target = args.use_delta_to_target
        self.rbf = args.rbf
        
        self.sigma = args.sigma

        G = args.dim
        # Per-atom features in aligned frame for the Transformer
        # pos_al(3), vel_al(3), delta_to_target(3 optional), distance(1)
        feat_dim = (2 * G) + (G if self.use_delta_to_target else 0) + 1

        self.input_proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Heads
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.vec_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, args.dim),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))
        #self.to(self.device)

    @staticmethod
    def _softplus_unit(x, beta=1.0, threshold=20.0, eps=1e-8):
        return F.softplus(x, beta=beta, threshold=threshold) + eps

    def forward(self, pos, vel, target):
        """
        pos, vel, target: (B,N,D)
        Returns: force (B,N,D), scale (B,N), vector (B,N,D)
        
        N: number of cells in batch
        D: dimension of gene vector
        """
        B, N, G = pos.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        # direction of target position
        delta = target - pos # (B,N,G)
        dist  = torch.norm(delta, dim=-1, keepdim=True)  # (B,N,1)
        feats = torch.cat([pos, vel, delta, dist], dim=-1) \
                if self.use_delta_to_target else torch.cat([pos, vel, dist], dim=-1)

        x = self.input_proj(feats) # (B,N,d_model)
        x = self.encoder(x) # (B,N,d_model)

        # Heads
        scale = self._softplus_unit(self.scale_head(x)).squeeze(-1)     # (B,N)
        vector = self.vec_head(x)                          # (B,N,3)

        # Direction field d
        d = (target - pos)

        # Parallel component from scale head
        scale = scale.unsqueeze(-1).expand(-1, -1, G) 
        scaled = scale * d # (B,N,3)

        # Project vector head output onto plane orthogonal to d
        eps = torch.finfo(pos.dtype).eps
        denom = d.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)        # (B,N,1)
        vec_parallel = ((vector * d).sum(dim=-1, keepdim=True) / denom) * d
        vec_perp = vector - vec_parallel

        return vec_perp + scaled 
    
class BiasForceTransformerNoVel(nn.Module):
    def __init__(self,
                 args,
                 d_model = 256,
                 nhead = 8,
                 num_layers = 4,
                 dim_feedforward = 512,
                 dropout = 0.1,
                 ):
        super().__init__()
        self.device = args.device
        self.N = args.num_particles
        
        self.use_delta_to_target = args.use_delta_to_target
        self.rbf = args.rbf
        
        self.sigma = args.sigma

        G = args.dim
        # Per-atom features in aligned frame for the Transformer
        # pos_al(3), vel_al(3), delta_to_target(3 optional), distance(1)
        feat_dim = G + (G if self.use_delta_to_target else 0) + 1

        self.input_proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Heads
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.vec_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, args.dim),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))
        #self.to(self.device)

    @staticmethod
    def _softplus_unit(x, beta=1.0, threshold=20.0, eps=1e-8):
        return F.softplus(x, beta=beta, threshold=threshold) + eps

    def forward(self, pos, target):
        """
        pos, target: (B,N,D)
        Returns: force (B,N,D), scale (B,N), vector (B,N,D)
        
        N: number of cells in batch
        D: dimension of gene vector
        """
        B, N, G = pos.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        # direction of target position
        delta = target - pos # (B,N,G)
        dist  = torch.norm(delta, dim=-1, keepdim=True)  # (B,N,1)
        feats = torch.cat([pos, delta, dist], dim=-1) \
                if self.use_delta_to_target else torch.cat([pos, dist], dim=-1)

        x = self.input_proj(feats) # (B,N,d_model)
        x = self.encoder(x) # (B,N,d_model)

        # Heads
        scale = self._softplus_unit(self.scale_head(x)).squeeze(-1)     # (B,N)
        vector = self.vec_head(x)                          # (B,N,3)

        # Direction field d
        d = (target - pos)

        # Parallel component from scale head
        scale = scale.unsqueeze(-1).expand(-1, -1, G) 
        scaled = scale * d # (B,N,3)

        # Project vector head output onto plane orthogonal to d
        eps = torch.finfo(pos.dtype).eps
        denom = d.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)        # (B,N,1)
        vec_parallel = ((vector * d).sum(dim=-1, keepdim=True) / denom) * d
        vec_perp = vector - vec_parallel

        return vec_perp + scaled 