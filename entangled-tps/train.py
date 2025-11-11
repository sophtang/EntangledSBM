import os
import sys
import argparse

import torch
import wandb

from entangled.mds import MDs
from entangled.utils.logging import Logger
from entangled.entangledsbm import EntangledSBM


def main():
    parser = argparse.ArgumentParser()
    # System Config
    parser.add_argument("--date", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--molecule", default="aldp", type=str)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--run_name', default=None, type=str)
    # Logger Config
    parser.add_argument("--save_dir", default="results", type=str)
    # Policy Config
    parser.add_argument("--bias", default="force", type=str)
    # Sampling Config
    parser.add_argument("--start_state", default="c5", type=str)
    parser.add_argument("--end_state", default="c7ax", type=str)
    parser.add_argument("--num_steps", default=1000, type=int)
    parser.add_argument("--timestep", default=1, type=float)
    parser.add_argument("--sigma", default=0.1, type=float)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--temperature", default=300, type=float)
    parser.add_argument("--friction", default=0.001, type=float)
    parser.add_argument("--rbf", action='store_true', default=False)
    parser.add_argument("--use_delta_to_target", action='store_true', default=False)
    # Training Config
    parser.add_argument("--start_temperature", default=600, type=float)
    parser.add_argument("--end_temperature", default=300, type=float)
    parser.add_argument("--num_rollouts", default=1000, type=int)
    parser.add_argument("--trains_per_rollout", default=1000, type=int)
    parser.add_argument("--log_z_lr", default=1e-3, type=float)
    parser.add_argument("--policy_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1, type=int)
    parser.add_argument("--control_variate", default="global", type=str)
    parser.add_argument("--self_normalize", action='store_true', default=False)
    parser.add_argument("--importance_sample", action='store_true', default=False)
    # path objective
    parser.add_argument("--objective", default="ce", type=str)
    parser.add_argument("--curriculum_rollouts", default=50, type=int)
    parser.add_argument("--sigma_min", default=0.25, type=float)
    parser.add_argument("--sigma_max", default=2.0, type=float)
    parser.add_argument("--resample_every", default=0, type=int)
    parser.add_argument("--resample_jitter", default=1e-3, type=float)
    parser.add_argument("--branch_beta", default=5.0, type=float)
    parser.add_argument("--bias_scale", default=1.0, type=float)
    parser.add_argument("--control_cost", action='store_true', default=False)
    parser.add_argument("--early_termination", action='store_true', default=False)
    parser.add_argument("--success_distance_threshold", default=1.0, type=float)
    parser.add_argument("--adaptive_bias", action='store_true', default=False)
    parser.add_argument("--min_bias_scale", default=0.5, type=float)
    parser.add_argument("--max_bias_scale", default=3.0, type=float)
    parser.add_argument("--target_dist_threshold", default=2.0, type=float)
    parser.add_argument("--vel_conditioned", action='store_true', default=False)
    
    args = parser.parse_args()
    
    
    args.training = True
    args.save_dir = args.save_dir
    
    positions_dir = f"{args.save_dir}/positions"
    if not os.path.exists(positions_dir):
        os.makedirs(positions_dir)
            
    if args.wandb:
        wandb.init(project="entangled-tps", config=args, name=args.run_name)
    torch.manual_seed(args.seed)
    mds = MDs(args)
    logger = Logger(args, mds)
    
    target_pos = mds.target_position.squeeze(0)
    
    model = EntangledSBM(args, mds)
    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )
    
    for rollout in range(args.num_rollouts):
        model.sample(args, mds, temperatures[rollout])
        loss, positions = model.train(args, mds)

        model.increment_rollout()

        logger(loss, rollout, model.bias_net)
        
        logger.plot()


if __name__ == "__main__":
    main()