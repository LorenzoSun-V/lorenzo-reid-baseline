# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import os
import argparse
from pathlib import Path
from prune import iterative_prune, once_iterative_reanalyse, once_iterative_xilinx, once_onestep
from quantize import quantize, once_quantize


def default_argument_parser():
    """
    Create a parser with some common arguments used.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Lorenzo ReID Baseline Training")
    parser.add_argument("-f", "--config-file", default="./configs/Market1501/bagtricks_R50.yml", type=str, help="path to config file")
    parser.add_argument("-m", "--mode", default="iter", choices=["iter", "nas", "ofa"], help="pruning method")
    parser.add_argument("--num-subnet", type=int, default=40, help="number of subnet to search, only useful in one_step prune")
    parser.add_argument("-s", "--sparsity-ratios", type=float, nargs='+', default=[0.4, 0.5, 0.6, 0.7], help="list of sparsity ratios")
    parser.add_argument("-d", "--gpus", type=str, default="0", help="string of gpus, like '0' ")
    return parser


def auto_machine(args):
    for index, sparsity_ratio in enumerate(args.sparsity_ratios):
        if args.mode == "iter":
            once_iterative_xilinx(args, index, sparsity_ratio)
        elif args.mode == "nas":
            once_onestep(args, sparsity_ratio)
        elif args.mode == "ofa":
            pass
        else:
            raise RuntimeError('''only support mode in ["iter", "nas", "ofa]!!! Please double check --mode in your command.''')
        once_quantize(args, sparsity_ratio)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    auto_machine(args)