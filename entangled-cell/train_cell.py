import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import wandb

from entangledcell_module_unseen import EntangledNetTrainCellUnseen
from entangledcell_module_three import EntangledNetTrainCellThree

# cell
from dataloaders.three_branch_data import ThreeBranchTahoeDataModule
from dataloaders.clonidine_v2_data import ClonidineV2DataModule

from geo_metrics.metric_factory import DataManifoldMetric
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torchcfm.optimal_transport import OTPlanSampler
from parser import parse_args
from train_utils import load_config, merge_config
from bias import BiasForceTransformer, BiasForceTransformerNoVel

def main():
    
    args = parse_args()
    if args.config_path:
        config = load_config(args.config_path)
        args = merge_config(args, config)
    
    args.training = True
    args.save_dir = args.save_dir
    
    # Create positions directory for saving trajectory samples
    positions_dir = f"{args.save_dir}/positions"
    if not os.path.exists(positions_dir):
        os.makedirs(positions_dir)
            
    wandb.init(project="entangled-cell", 
                config=args, 
                name=args.run_name)
        
    torch.manual_seed(args.seed)
    
    ot_sampler = (
        OTPlanSampler(method=args.optimal_transport_method)
        if args.optimal_transport_method != "None"
        else None
    )
    
    # get data
    if args.data_name == "trametinib":
        datamodule = ThreeBranchTahoeDataModule(args=args)
    else:
         datamodule = ClonidineV2DataModule(args=args)
        
    # data manifold metrics
    data_manifold_metric = DataManifoldMetric(
        args=args,
        skipped_time_points=[],
        datamodule=datamodule,
    )
    
    if args.vel_conditioned:
        bias_net = BiasForceTransformer(args)
    else:
        print("Using no velocity conditioned model")
        bias_net = BiasForceTransformerNoVel(args)
    
    timepoint_data = datamodule.get_timepoint_data()
    
    if args.data_name == "trametinib":
        entangled_train = EntangledNetTrainCellThree(args=args,
                                            bias_net=bias_net,
                                            data_manifold_metric=data_manifold_metric,
                                            timepoint_data=timepoint_data,
                                            ot_sampler=ot_sampler,
                                            vel_conditioned=args.vel_conditioned)
    else:
        entangled_train = EntangledNetTrainCellUnseen(args=args,
                                            bias_net=bias_net,
                                            data_manifold_metric=data_manifold_metric,
                                            timepoint_data=timepoint_data,
                                            ot_sampler=ot_sampler,
                                            vel_conditioned=args.vel_conditioned)
    
    wandb_logger = WandbLogger()
    
    trainer = Trainer(
        max_epochs=args.num_rollouts,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir=args.root_dir,
        gradient_clip_val=None,
        devices=[0],
    )
    
    trainer.fit(
        entangled_train, datamodule=datamodule
    )
    trainer.test(entangled_train, datamodule=datamodule)

if __name__ == "__main__":
    main()