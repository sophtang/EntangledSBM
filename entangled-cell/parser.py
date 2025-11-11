import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, 
        default='', 
        help="Path to config file"
    )
    parser.add_argument(
        "--optimal_transport_method",
        type=str,
        default="exact",
        help="Use optimal transport in CFM training",
    )
    parser.add_argument(
        "--split_ratios",
        nargs=2,
        type=float,
        default=[0.9, 0.1],
        help="Split ratios for training/validation data in CFM training",
    )
    parser.add_argument(
        "--accelerator", type=str, default="cpu", help="Training accelerator"
    )
    parser.add_argument("--date", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--molecule", default="aldp", type=str)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--unseen', action='store_true', default=False)
    parser.add_argument('--run_name', default=None, type=str)
    # Logger Config
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--root_dir", default="", type=str)
    # Policy Config
    parser.add_argument("--bias", default="force", type=str)
    # Sampling Config
    parser.add_argument("--start_state", default="c5", type=str)
    parser.add_argument("--end_state", default="c7ax", type=str)
    parser.add_argument("--num_steps", default=100, type=int)
    #parser.add_argument("--timestep", default=1, type=float)
    parser.add_argument("--sigma", default=0.1, type=float)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--temperature", default=300, type=float)
    parser.add_argument("--friction", default=2.0, type=float)
    parser.add_argument("--rbf", action='store_true', default=False)
    parser.add_argument("--use_delta_to_target", action='store_true', default=False)
    parser.add_argument("--use_gnn", action='store_true', default=False)
    # Training Config
    parser.add_argument("--start_temperature", default=600, type=float)
    parser.add_argument("--end_temperature", default=300, type=float)
    parser.add_argument("--num_rollouts", default=1000, type=int)
    parser.add_argument("--trains_per_rollout", default=1000, type=int)
    parser.add_argument("--log_z_lr", default=1e-3, type=float)
    parser.add_argument("--policy_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1, type=int)
    parser.add_argument("--control_variate", default="global", type=str)
    parser.add_argument("--self_normalize", action='store_true', default=False)
    # path objective
    parser.add_argument("--objective", default="ce", type=str)
    parser.add_argument("--vel_conditioned", action='store_true', default=False)
    parser.add_argument("--dir_only", action='store_true', default=False)
    
    # cell experiment
    parser.add_argument("--num_particles", default=16, type=int)
    #parser.add_argument("--gene_dim", default=50, type=int)
    parser.add_argument("--kT", type=float, default=0.0)
    ######### DATASETS #################
    parser = datasets_parser(parser)

    ######### METRICS ##################
    parser = metric_parser(parser)

    return parser.parse_args()


def datasets_parser(parser):
    parser.add_argument("--dim", type=int, default=50, help="Dimension of data")

    parser.add_argument(
        "--data_type",
        type=str,
        default="tahoe",
        help="Type of data, now wither scrna or one of toys",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="tahoe",
        help="Path to the dataset",
    )
    return parser


def metric_parser(parser):
    parser.add_argument(
        "--n_centers",
        type=int,
        default=300,
        help="Number of centers for RBF network",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.5,
        help="Kappa parameter for RBF network",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=-2.75,
        help="Rho parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--velocity_metric",
        type=str,
        default="rbf",
        help="Metric for velocity calculation",
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=0.2,
        help="Gamma parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--metric_epochs",
        type=int,
        default=200,
        help="Number of epochs for metric learning",
    )
    parser.add_argument(
        "--metric_patience",
        type=int,
        default=25,
        help="Patience for metric learning",
    )
    parser.add_argument(
        "--metric_lr",
        type=float,
        default=1e-2,
        help="Learning rate for metric learning",
    )
    parser.add_argument(
        "--alpha_metric",
        type=float,
        default=1.0,
        help="Alpha parameter for metric learning",
    )
    return parser

