import torch
import joblib
import mdtraj as md
import pyemma.coordinates as coor


def pairwise_dist(x):
    dist_matrix = torch.cdist(x, x)
    return dist_matrix


def kabsch(P, Q):
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
    Vt[d < 0.0, -1] *= -1.0
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))
    return R, t

# safe for gradient computation
def kabsch_safe(P: torch.Tensor, T: torch.Tensor):
    """
    P, T: (..., N, 3)
    Returns R (...,3,3), t (...,1,3)
    """
    p_centroid = P.mean(dim=-2, keepdim=True)
    t_centroid = T.mean(dim=-2, keepdim=True)

    P0 = P - p_centroid
    T0 = T - t_centroid

    H = P0.transpose(-2, -1) @ T0                   # (...,3,3)
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V  = Vh.transpose(-2, -1)                       # out-of-place
    Ut = U.transpose(-2, -1)

    # Reflection correction WITHOUT any in-place on U/V/Vh
    det = torch.det(V @ Ut)                         # (...,)
    s = torch.where(det < 0, -1.0, 1.0)             # (...,)
    ones = torch.ones_like(s)
    F = torch.diag_embed(torch.stack([ones, ones, s], dim=-1))  # (...,3,3)

    R = V @ F @ Ut                                   # (...,3,3)
    t = t_centroid - p_centroid @ R.transpose(-2, -1)# (...,1,3)
    return R, t

def compute_dihedral(positions):
    v = positions[:, :-1] - positions[:, 1:]
    v0 = -v[:, 0]
    v1 = v[:, 2]
    v2 = v[:, 1]
    s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )
    s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )
    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2
    v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    x = torch.sum(v0 * v1, dim=-1)
    v3 = torch.cross(v0, v2, dim=-1)
    y = torch.sum(v3 * v1, dim=-1)
    return torch.atan2(y, x)


def aldp_diff(position, target_position):
    angle_2 = torch.tensor([1, 6, 8, 14], dtype=torch.long, device=position.device)
    angle_1 = torch.tensor([6, 8, 14, 16], dtype=torch.long, device=position.device)
    target_psi = compute_dihedral(target_position[:, angle_1])
    target_phi = compute_dihedral(target_position[:, angle_2])
    psi = compute_dihedral(position[:, angle_1])
    phi = compute_dihedral(position[:, angle_2])
    psi_diff = torch.abs(psi - target_psi) % (2 * torch.pi)
    psi_diff = torch.min(psi_diff, 2 * torch.pi - psi_diff)
    phi_diff = torch.abs(phi - target_phi) % (2 * torch.pi)
    phi_diff = torch.min(phi_diff, 2 * torch.pi - phi_diff)
    return psi_diff, phi_diff


def tic_diff(molecule, position, target_position):
    tica_model = joblib.load(f"./data/{molecule}/tica_model.pkl")
    feat = coor.featurizer(f"./data/{molecule}/folded.pdb")
    feat.add_backbone_torsions(cossin=True)
    traj = md.Trajectory(
        target_position.cpu().numpy(),
        md.load(f"./data/{molecule}/folded.pdb").topology,
    )
    feature = feat.transform(traj)
    tica_target = tica_model.transform(feature)
    tica_target = torch.from_numpy(tica_target).to(position.device)
    traj = md.Trajectory(
        position.cpu().numpy(),
        md.load(f"./data/{molecule}/folded.pdb").topology,
    )
    feature = feat.transform(traj)
    tica = tica_model.transform(feature)
    tica = torch.from_numpy(tica).to(position.device)
    tic1_diff = abs(tica[:, 0] - tica_target[:, 0])
    tic2_diff = abs(tica[:, 1] - tica_target[:, 1])
    return tic1_diff, tic2_diff
