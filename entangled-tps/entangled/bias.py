import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import kabsch
from .utils.rbf import grad_log_wrt_positions

class BiasForceTransformer(nn.Module):
    def __init__(self,
                 mds,
                 args,
                 d_model = 256,
                 nhead = 8,
                 num_layers = 4,
                 dim_feedforward = 512,
                 dropout = 0.1,
                 ):
        super().__init__()
        self.device = args.device
        self.heavy_atoms = mds.heavy_atoms
        self.N = mds.num_particles
        
        self.use_delta_to_target = args.use_delta_to_target
        self.rbf = args.rbf
        
        self.sigma = args.sigma

        feat_dim = 3 + 3 + (3 if self.use_delta_to_target else 0) + 1

        self.input_proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.vec_head_aligned = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        
        self.bias = args.bias

        self.log_z = nn.Parameter(torch.tensor(0.0))
        self.to(self.device)

    @staticmethod
    def _softplus_unit(x, beta=1.0, threshold=20.0, eps=1e-8):
        return F.softplus(x, beta=beta, threshold=threshold) + eps

    def forward(self, pos, vel, target):
        """
        pos, vel, target: (B,N,3)
        Returns: force (B,N,3), scale (B,N), vector (B,N,3)
        """
        B, N, _ = pos.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        heavy = self.heavy_atoms.to(pos.device)

        pos_h, tgt_h = pos[:, heavy], target[:, heavy]  # (B,Nh,3)
        R, t = kabsch(pos_h, tgt_h)

        pos_al = pos @ R.transpose(-2, -1) + t
        vel_al = vel @ R.transpose(-2, -1)

        delta_al = target - pos_al # (B,N,3)
        dist_al  = torch.norm(delta_al, dim=-1, keepdim=True)  # (B,N,1)
        feats = torch.cat([pos_al, vel_al, delta_al, dist_al], dim=-1) \
                if self.use_delta_to_target else torch.cat([pos_al, vel_al, dist_al], dim=-1)

        x = self.input_proj(feats) # (B,N,d_model)
        x = self.encoder(x) # (B,N,d_model)

        scale = self._softplus_unit(self.scale_head(x)).squeeze(-1) # (B,N)
        vec_aligned = self.vec_head_aligned(x) # (B,N,3)

        vector = vec_aligned @ R  # (B,N,3)

        target_posframe = (target - t) @ R # (B,N,3)

        if self.rbf:
            d = grad_log_wrt_positions(pos, target_posframe, self.sigma).detach()
        else:
            d = (target_posframe - pos)

        scale = scale.unsqueeze(-1).expand(-1, -1, 3) 
        scaled = scale * d 

        eps = torch.finfo(pos.dtype).eps
        denom = d.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps) # (B,N,1)
        vec_parallel = ((vector * d).sum(dim=-1, keepdim=True) / denom) * d
        vec_perp = vector - vec_parallel

        return vec_perp + scaled 

class BiasForceTransformerNoVel(nn.Module):
    def __init__(self,
                 mds,
                 args,
                 d_model = 256,
                 nhead = 8,
                 num_layers = 4,
                 dim_feedforward = 512,
                 dropout = 0.1,
                 ):
        super().__init__()
        self.device = args.device
        self.heavy_atoms = mds.heavy_atoms
        self.N = mds.num_particles
        
        self.use_delta_to_target = args.use_delta_to_target
        self.rbf = args.rbf
        
        self.sigma = args.sigma

        feat_dim = 3 + (3 if self.use_delta_to_target else 0) + 1

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
        self.vec_head_aligned = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))
        self.to(self.device)

    @staticmethod
    def _softplus_unit(x, beta=1.0, threshold=20.0, eps=1e-8):
        return F.softplus(x, beta=beta, threshold=threshold) + eps

    def forward(self, pos, target):
        """
        pos, target: (B,N,D)
        Returns: force (B,N,D), scale (B,N), vector (B,N,D)
        
        N: number of atoms
        D: dimension (3)
        """
        B, N, _ = pos.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        heavy = self.heavy_atoms.to(pos.device)
        
        pos_h, tgt_h = pos[:, heavy], target[:, heavy]  # (B,Nh,3)
        R, t = kabsch(pos_h, tgt_h)

        pos_al = pos @ R.transpose(-2, -1) + t

        delta_al = target - pos_al # (B,N,3)
        dist_al  = torch.norm(delta_al, dim=-1, keepdim=True)  # (B,N,1)
        feats = torch.cat([pos_al, delta_al, dist_al], dim=-1) \
                if self.use_delta_to_target else torch.cat([pos_al, dist_al], dim=-1)

        x = self.input_proj(feats) # (B,N,d_model)
        x = self.encoder(x) # (B,N,d_model)

        # Heads
        scale = self._softplus_unit(self.scale_head(x)).squeeze(-1)  # (B,N)
        vec_aligned = self.vec_head_aligned(x) # (B,N,3)

        vector = vec_aligned @ R  # (B,N,3)

        target_posframe = (target - t) @ R  
        
        if self.rbf:
            d = grad_log_wrt_positions(pos, target_posframe, self.sigma).detach()
        else:
            d = (target_posframe - pos)

        scale = scale.unsqueeze(-1).expand(-1, -1, 3) 
        scaled = scale * d # (B,N,3)

        eps = torch.finfo(pos.dtype).eps
        denom = d.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps) # (B,N,1)
        vec_parallel = ((vector * d).sum(dim=-1, keepdim=True) / denom) * d
        vec_perp = vector - vec_parallel

        return vec_perp + scaled 