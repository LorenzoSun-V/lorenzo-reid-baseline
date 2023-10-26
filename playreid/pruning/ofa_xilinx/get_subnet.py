# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import argparse
from collections import OrderedDict
import os
import sys
import torch
from pytorch_nndct import OFAPruner

sys.path.append('.')
from config.config import get_cfg
from playreid.data import build_reid_train_loader
from playreid.engine.engine import get_evaluator, auto_scale_hyperparams, default_setup
from playreid.evaluation import print_csv_format, inference_on_dataset
from playreid.modeling import build_model
from playreid.utils.checkpoint import Checkpointer


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
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'ofa_prune')
    cfg.freeze()
    default_setup(cfg, args, 'log_get_subnet.txt')
    return cfg

def get_gpus(device):
    return [int(i) for i in device.split(',')]



def main():
    parser = argparse.ArgumentParser(description="Lorenzo ReID Baseline Pruning")
    parser.add_argument(
        "--config-file", default="/workspace/lorenzo/ReID/lorenzo-reid-baseline/configs/Market1501/bagtricks_R50.yml", 
        help="path to config file", type=str)
    parser.add_argument("--gpus", type=str, default='0', help="String of available GPU number")
    parser.add_argument("--sparsity", type=float, default=0.5, help="sparsity ratio of model")
    parser.add_argument("--last-sparsity", type=float, default=0, help="sparsity ratio of model last time")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = setup(args)
    gpus = get_gpus(args.gpus)
    
    # init reid model & load trained pth
    train_loader = build_reid_train_loader(cfg)
    cfg = auto_scale_hyperparams(cfg, train_loader.dataset.num_classes)
    model = build_model(cfg)
    if args.last_sparsity == 0:
        model_path = os.path.join(os.path.dirname(cfg.OUTPUT_DIR), 'model_best.pth')
    else:
        model_path = os.path.join(cfg.OUTPUT_DIR, f"sparsity_{args.last_sparsity}/model_sparsity_{args.last_sparsity}.pth")
    Checkpointer(model).load(model_path)

    input_signature = torch.randn([1, 3, 256, 128], dtype=torch.float32)
    input_signature = input_signature.to(torch.device(cfg.MODEL.DEVICE))


if __name__ == '__main__':
    main()