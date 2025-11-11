import sys
import torch
import numpy as np
from tqdm import tqdm

import math
from .utils.utils import kabsch
from .bias import BiasForceTransformer


class EntangledSBM:
    def __init__(self, args, mds):
        self.bias_net = BiasForceTransformer(mds, args)
        
        self.target_measure = PathObjective(args, mds)
            
        if args.training:
            self.replay = ReplayBuffer(args, mds)
        
        self.rollout_idx = 0
    
    def increment_rollout(self):
        self.rollout_idx += 1
    

    def sample(self, args, mds, temperature):
        
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        
        forces = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        
        position, force = mds.report()
        positions[:, 0] = position.detach().clone()
        forces[:, 0] = force.detach().clone()
        mds.reset()
        
        mds.set_temperature(temperature)
        prev_position = position.detach().clone()
        
        for step in tqdm(range(1, args.num_steps + 1), desc="Sampling"):
            if step == 1:
                velocity = torch.zeros_like(position)
            else:
                velocity = (position - prev_position) / args.timestep
            
            bias_force = self.bias_net(position.detach().clone(), 
                                       velocity.detach().clone(), 
                                       mds.target_position).detach()

            
            mds.step(bias_force)
            
            position, force = mds.report()
            
            if not _is_finite(position, force):
                print("MD produced non-finite: pos nan/inf", torch.isnan(position).sum().item(), torch.isinf(position).sum().item(),
                    "force nan/inf", torch.isnan(force).sum().item(), torch.isinf(force).sum().item())

                positions[:, step] = prev_position
                forces[:, step] = force
                break
            
            prev_position = position.detach().clone()
            
            positions[:, step] = position
            forces[:, step] = force - 1e-6 * bias_force # kJ/(mol*nm) -> (da*nm)/fs**2
                
        mds.reset()
        log_tpm, final_idx, log_ri = self.target_measure(positions, forces)
        
        if args.training:
            self.replay.add_ranked((positions, 
                                    forces, 
                                    log_tpm), score=log_ri)
            
        for i in range(args.num_samples):
            np.save(
                f"{args.save_dir}/positions/{i}.npy",
                positions[i][: final_idx[i] + 1].cpu().numpy(),
            )

    def train(self, args, mds):
        
        exclude = {id(self.bias_net.log_z)}
        params_except = [p for p in self.bias_net.parameters() if id(p) not in exclude]
        optimizer = torch.optim.Adam(
            [
                {"params": [self.bias_net.log_z], "lr": args.log_z_lr},
                {"params": params_except, "lr": args.policy_lr},
            ]
        )
        loss_sum = 0
        
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            
            positions, forces, log_tpm, log_ri = self.replay.sample()
            
            assert positions.shape == forces.shape, f"{positions.shape=} != {forces.shape=}"
            velocities = (positions[:, 1:] - positions[:, :-1]) / args.timestep
            
            
            biases = 1e-6 * self.bias_net(
                positions[:, :-1].reshape(-1, positions.size(-2), positions.size(-1)),
                velocities.view(-1, velocities.size(-2), velocities.size(-1)), # should this be forces or velocities?
                mds.target_position,
            )

            biases = biases.view(*velocities.shape)
            
            means = (
                1 - args.friction * args.timestep
            ) * velocities + args.timestep / mds.m * (forces[:, :-1] + biases)

            resid = _sanitize(velocities[:, 1:] - means[:, :-1])
            log_bpm = mds.log_prob(resid).mean((1, 2, 3))
            
            if args.control_variate == "global":
                log_z = self.bias_net.log_z
            elif args.control_variate == "local":
                log_z = (log_tpm - log_bpm).mean().detach()
            elif args.control_variate == "zero":
                log_z = 0
            
            # compute loss
            if args.objective == "ce": # cross entropy
                
                log_rnd = (log_tpm - log_bpm).detach()
                
                weights = torch.softmax(log_rnd, dim=0)
                
                if args.control_cost:
                    control_cost = 0.5 * args.timestep * (biases[:, :-1].square().sum((-1, -2, -3))).mean()
                    loss = -(weights * log_bpm).sum() + control_cost
                else:
                    loss = -(weights * log_bpm).sum()
                
            elif args.objective == "lv": # log-variance
                loss = (log_z + log_bpm - log_tpm).square().mean()

            loss.backward()
            
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)
                
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            
        loss = loss_sum / args.trains_per_rollout
        return loss, positions


class ReplayBuffer:
    def __init__(self, args, mds):
        
        self.positions = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.forces = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.log_tpm = torch.zeros(args.buffer_size, device=args.device)
        self.idx = 0
        
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.buffer_size = args.buffer_size
        self.args = args
        
        # new
        self.scores = torch.zeros(args.buffer_size, device=args.device)
        self.count = 0

    def add(self, data):
        pos_batch, force_batch, tpm_batch = data

        newN = pos_batch.size(0)

        indices = (torch.arange(self.idx, self.idx + newN, device=self.device) % self.buffer_size)

        self.idx = (self.idx + newN) % self.buffer_size

        self.positions[indices] = pos_batch.detach().to(self.device).clone()
        self.forces[indices] = force_batch.detach().to(self.device).clone()
        self.log_tpm[indices] = tpm_batch.detach().to(self.device).clone()

        self.count = min(self.count + newN, self.buffer_size)
        
    @torch.no_grad()
    def add_ranked(self, data, score=None):
        positions, forces, log_tpm = data
        if score is None:
            score = log_tpm
    
        # detach to avoid holding graphs
        positions, forces, log_tpm, score = (
            positions.clone().detach(), 
            forces.clone().detach(), 
            log_tpm.clone().detach(), 
            score.clone().detach()
        )
        
        valid = torch.isfinite(positions).all((1,2,3)) & torch.isfinite(forces).all((1,2,3)) & torch.isfinite(log_tpm)

        if valid.any():
            positions = positions[valid]
            forces = forces[valid]
            log_tpm = log_tpm[valid]
            score = score[valid]

            curr = self.count
            newN = positions.size(0)
            keepN = min(self.buffer_size, curr + newN)

            if curr > 0:
                pos_cat = torch.cat([self.positions[:curr], positions], dim=0)
                force_cat = torch.cat([self.forces[:curr], forces],    dim=0)
                tpm_cat = torch.cat([self.log_tpm[:curr],  log_tpm],   dim=0)
                sco_cat = torch.cat([self.scores[:curr],   score],     dim=0)
            else:
                pos_cat, force_cat, tpm_cat, sco_cat = positions, forces, log_tpm, score

            top_vals, top_idx = torch.topk(sco_cat, k=keepN, largest=True, sorted=False)

            self.positions[:keepN] = pos_cat.index_select(0, top_idx)
            self.forces[:keepN] = force_cat.index_select(0, top_idx)
            self.log_tpm[:keepN] = tpm_cat.index_select(0, top_idx)
            self.scores[:keepN] = top_vals
            self.count = keepN

    def sample(self):
        assert self.count > 0, "buffer is empty"
        if self.args.importance_sample:
            idx = torch.multinomial(torch.softmax(self.scores[:self.count], 0),
                        num_samples=self.batch_size, replacement=True)
        else: 
            idx = torch.randint(0, self.count, (self.batch_size,), device=self.device)
        # Return detached clones so callers can modify tensors or backprop without
        # affecting the stored buffer or retaining autograd graphs.
        return (
            self.positions[idx].clone().detach(),
            self.forces[idx].clone().detach(),
            self.log_tpm[idx].clone().detach(),
            self.scores[idx].clone().detach(),
        )


class PathObjective:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.timestep = args.timestep
        self.friction = args.friction
        self.heavy_atoms = mds.heavy_atoms
        self.target_position = mds.target_position
        self.m = mds.m
        self.log_prob = mds.log_prob

    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri, final_idx = self.relaxed_indicator(positions, self.target_position)
        log_tpm = log_upm + log_ri
        return log_tpm, final_idx, log_ri

    def unbiased_path_measure(self, positions, forces):
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        
        means = (
            1 - self.friction * self.timestep
        ) * velocities + self.timestep / self.m * forces[:, :-1]
        
        resid = _sanitize(velocities[:, 1:] - means[:, :-1])
        
        lp = self.log_prob(resid)
        
        log_upm = lp.mean((1, 2, 3))
        return log_upm

    def relaxed_indicator(self, positions, target_position):
        positions = positions[:, :, self.heavy_atoms]
        target_position = target_position[:, self.heavy_atoms]
        
        log_ri = torch.zeros(positions.size(0), device=positions.device)
        final_idx = torch.zeros(
            positions.size(0), device=positions.device, dtype=torch.long
        )
        for i in range(positions.size(0)):
            log_ri[i], final_idx[i] = self.rbf(
                positions[i],
                target_position,
            ).max(0)
        return log_ri, final_idx

    def rbf(self, positions, target_position):
        
        R, t = kabsch(positions, target_position)
        positions = torch.matmul(positions, R.transpose(-2, -1)) + t
        log_ri = (
            -0.5 / self.sigma**2 * (positions - target_position).square().mean((-2, -1))
        )
        return log_ri
    
def _is_finite(*tensors):
    return all(torch.isfinite(t).all().item() for t in tensors)

def _sanitize(t, max_abs=1e6):
    t = torch.nan_to_num(t, nan=0.0, posinf=max_abs, neginf=-max_abs)
    return torch.clamp(t, min=-max_abs, max=max_abs)