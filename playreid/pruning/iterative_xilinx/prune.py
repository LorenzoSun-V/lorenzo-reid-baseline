# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import argparse
import logging
import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
from pytorch_nndct import get_pruning_runner
sys.path.append('.')
from config.config import get_cfg
from playreid.data import build_reid_train_loader, _train_loader_from_config
from playreid.engine.engine import do_test, auto_scale_hyperparams, default_setup
from playreid.engine.lanuch import launch
from playreid.modeling import build_model
from playreid.utils.checkpoint import Checkpointer


def default_argument_parser():
    """
    Create a parser with some common arguments used.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Lorenzo ReID Baseline Training")
    parser.add_argument("-f", "--config-file", default="/workspace/lorenzo/ReID/lorenzo-reid-baseline/configs/Market1501/bagtricks_R50.yml", type=str, help="path to config file")
    parser.add_argument("-s", "--sparsity", type=float, default=0.5, help="sparsity ratio of model")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser

    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    # This function loads default configuration written in ./config/default.py .
    cfg = get_cfg()
    # Merge configs from a given yaml file.
    cfg.merge_from_file(args.config_file)
    # Merge configs from list generated by args.opts
    cfg.merge_from_list(args.opts)  # ['MODEL.DEVICE', 'cuda:0']
    cfg.MODEL.PRETRAIN = False
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'iterative_prune', f'sparsity_{args.sparsity}')
    cfg.freeze()
    default_setup(cfg, args, 'log_prune.txt')
    return cfg


def main(args):
    cfg = setup(args)
    # train_loader = build_reid_train_loader(cfg)
    cfg = auto_scale_hyperparams(cfg, _train_loader_from_config(cfg)['train_set'].num_classes)
    model = build_model(cfg)
    sparse_model_path = os.path.join(cfg.OUTPUT_DIR, f"model_sparsity_{args.sparsity}.pth")
    Checkpointer(model).load(sparse_model_path)
    input_signature = torch.randn([1, 3, 256, 128], dtype=torch.float32)
    input_signature = input_signature.to(torch.device(cfg.MODEL.DEVICE))
    pruning_runner = get_pruning_runner(model, input_signature, 'iterative')
    slim_model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim',
                                      excludes=[model.backbone.layer4[2].conv3,
                                                model.heads])
    Checkpointer(slim_model, cfg.OUTPUT_DIR).save(f"model_slim_{args.sparsity}")
    # print(slim_model)
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)