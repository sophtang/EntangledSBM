import sys
import torch
from tqdm import tqdm
from torch.distributions import Normal

from .dynamics import dynamics
from .utils.utils import kabsch


class MDs:
    def __init__(self, args):
        self.device = args.device
        self.molecule = args.molecule
        self.end_state = args.end_state
        self.num_samples = args.num_samples
        self.start_state = args.start_state
        self.get_md_info(args)
        self.mds = self._init_mds(args)
        
        self.log_prob = Normal(0, self.std).log_prob
        self.target_position = self.target_position - self.target_position[:, self.heavy_atoms].mean(-2, keepdim=True)
        R, t = kabsch(
            self.start_position[:, self.heavy_atoms],
            self.target_position[:, self.heavy_atoms],
        )
        self.start_position = torch.matmul(self.start_position, R.transpose(-2, -1)) + t

    def get_md_info(self, args):
        md = getattr(dynamics, self.molecule.title())(args, self.end_state)
        self.num_particles = md.num_particles
        self.heavy_atoms = torch.from_numpy(md.heavy_atoms).to(self.device)
        self.energy_function = md.energy_function
        self.target_position = torch.tensor(
            md.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.std = torch.tensor(
            md.std,
            dtype=torch.float,
            device=args.device,
        )
        self.m = torch.tensor(
            md.m,
            dtype=torch.float,
            device=args.device,
        ).unsqueeze(-1)

    def _init_mds(self, args):
        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = getattr(dynamics, self.molecule.title())(args, self.start_state)
            mds.append(md)
        self.start_position = torch.tensor(
            md.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        return mds

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        positions, forces = [], []
        for i in range(self.num_samples):
            position, force = self.mds[i].report()
            positions.append(position)
            forces.append(force)
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        forces = torch.tensor(forces, dtype=torch.float, device=self.device)
        return positions, forces

    def reset(self):
        for i in range(self.num_samples):
            self.mds[i].reset()

    def set_temperature(self, temperature):
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)
