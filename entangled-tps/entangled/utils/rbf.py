import torch

from .utils import kabsch
    
def rbf(positions, target_position, sigma):
    R, t = kabsch(positions.detach(), target_position.detach())
    positions = torch.matmul(positions, R.transpose(-2, -1)) + t
    log_ri = (
        -0.5 / sigma**2 * (positions - target_position).square().mean((-2, -1))
    )
    return log_ri

def grad_log_wrt_positions(positions, target_position, sigma):
    """
    Gradient of log kernel w.r.t. the ORIGINAL positions: same shape as positions (..., N, 3).
    """
    pos = positions.clone().detach().requires_grad_(True)
    log_ri = rbf(pos, target_position, sigma)
    # sum over batch dims if present to get a scalar for autograd
    (grad_pos,) = torch.autograd.grad(log_ri.sum(), pos, create_graph=False, retain_graph=False)
    return grad_pos