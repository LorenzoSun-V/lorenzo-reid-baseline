# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import os
import argparse


def default_argument_parser():
    """
    Create a parser with some common arguments used.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Lorenzo ReID Baseline Training")
    parser.add_argument("-f", "--config-file", default="./configs/Market1501/bagtricks_R50.yml", type=str, help="path to config file")
    parser.add_argument("-s", "--sparsity-ratios", type=float, nargs='+', default=[0.3, 0.5, 0.7], help="list of sparsity ratios")
    parser.add_argument("-m", "--mode", default="iter", choices=["iter", "nas", "ofa"], help="pruning method")
    return parser


def once_quantize(args, sparsity_ratio):
    # 1. qat train
    qat_train_sh = f"python3 playreid/quantizing/quantize_xilinx/qat.py \
                            -f {args.config_file} \
                            -d 1 \
                            -s {sparsity_ratio} \
                            -m {args.mode} \
                            -q 'train'\
                            SOLVER.IMS_PER_BATCH 128 \
                            SOLVER.BASE_LR 7e-5 \
                            SOLVER.STEPS 25,65,95"
    # os.system(qat_train_sh)

    # 2. qat deploy
    qat_deploy_sh = f"python3 playreid/quantizing/quantize_xilinx/qat.py \
                            -f {args.config_file} \
                            -d 1 \
                            -s {sparsity_ratio} \
                            -m {args.mode} \
                            -q 'deploy'" 
    # os.system(qat_deploy_sh)

    # 3. extract bn parameters
    extract_bn_param = f" python3 playreid/quantizing/quantize_xilinx/extract_bn_params.py \
                                -f {args.config_file} \
                                -m {args.mode} \
                                -s {sparsity_ratio}"
    # os.system(extract_bn_param)

    # 4. qat test(optional)
    qat_test_sh = f"python3 playreid/quantizing/quantize_xilinx/qat.py \
                            -f {args.config_file} \
                            -d 1 \
                            -s {sparsity_ratio} \
                            -m {args.mode} \
                            -q 'test'" 
    os.system(qat_test_sh)


def quantize(args):
    for sparsity_ratio in args.sparsity_ratios:
        once_quantize(args, sparsity_ratio)
   

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    quantize(args)