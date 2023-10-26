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
from pytorch_nndct import get_pruning_runner
from pytorch_nndct import QatProcessor
if os.environ["W_QUANT"]=='1':
    import pytorch_nndct
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

sys.path.append('.')
from config.config import get_cfg
from playreid.engine.engine import default_setup
from playreid.modeling import build_model
from playreid.quantizing.quantize_xilinx.eval_quantize import eval_quantize
from playreid.utils.checkpoint import Checkpointer
from playreid.utils.file_io import find_prune_folder


def default_argument_parser():
    """
    Create a parser with some common arguments used.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Lorenzo ReID Baseline Training")
    parser.add_argument("--config-file", default="/workspace/lorenzo/ReID/lorenzo-reid-baseline/configs/Market1501/bagtricks_R50.yml", type=str, help="path to config file")
    parser.add_argument("--sparsity", type=float, default=0.5, help="sparsity ratio of retrain")
    parser.add_argument("--mode", default="iter", choices=["iter", "nas", "ofa"], help="pruning method")
    parser.add_argument('--quant-mode', default='calib', choices=['float', 'calib', 'test'],
                        help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')    
    parser.add_argument("--fast-finetune", action="store_true", help="quantize with fast-finetuning")
    parser.add_argument("--dump-xmodel", action="store_true", help="dump xmodel after test")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser

    
def setup(args, folder_name):
    """
    Create configs and perform basic setups.
    """
    # This function loads default configuration written in ./config/default.py .
    cfg = get_cfg()
    # Merge configs from a given yaml file.
    cfg.merge_from_file(args.config_file)
    # Merge configs from list generated by args.opts
    cfg.merge_from_list(args.opts)  # ['MODEL.DEVICE', 'cuda:0']
    if args.dump_xmodel:
        cfg.TEST.IMS_PER_BATCH = 1
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.PRETRAIN = False
    cfg.MODEL.BACKBONE.PRETRAIN = False
    log_root_dir = cfg.OUTPUT_DIR
    if args.fast_finetune:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'quantizing', folder_name, f'sparsity_{args.sparsity}_finetune')
    else:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'quantizing', folder_name, f'sparsity_{args.sparsity}')
    cfg.freeze()
    default_setup(cfg, args, 'log_quantize.txt')
    return cfg, log_root_dir


def main(args):
    if args.sparsity != 0:
        folder_name = find_prune_folder(args.mode)
    else:
        folder_name = "original_pth"
    cfg, log_root_dir = setup(args, folder_name)
    model = build_model(cfg)
    # print(model)
    # return
    if args.sparsity != 0:
        slim_model_path = os.path.join(log_root_dir, folder_name, f"sparsity_{args.sparsity}", f"model_slim_{args.sparsity}.pth")
        assert os.path.exists(slim_model_path), f"{slim_model_path} No slim model!"
        input_signature = torch.randn([1, 3, 256, 128], dtype=torch.float32)
        input_signature = input_signature.to(torch.device(cfg.MODEL.DEVICE))
        prune_method = folder_name[:-6]
        pruning_runner = get_pruning_runner(model, input_signature, prune_method)
        if prune_method == "iterative":
            slim_model = pruning_runner.prune(ratio=args.sparsity, mode='slim')
        elif prune_method == "one_step":
            slim_model = pruning_runner.prune(mode='slim')
        Checkpointer(slim_model).load(slim_model_path)
    else:
        slim_model = model
        model_path = os.path.join(log_root_dir, "model_best.pth")
        Checkpointer(slim_model).load(model_path)
    slim_model.eval()
    # do_test(cfg, slim_model)
    # return 
    resize_wh = cfg.INPUT.SIZE_TEST
    x = torch.randn(32, 3, resize_wh[0],resize_wh[1]).to(torch.device(cfg.MODEL.DEVICE))
    if args.quant_mode == 'float':
        quant_model = slim_model
    else:
        # xmodel_output = os.path.join(cfg.OUTPUT_DIR, f'sparsity_{args.sparsity}')
        quantizer = torch_quantizer(args.quant_mode, slim_model, (x), output_dir=cfg.OUTPUT_DIR, device=torch.device(cfg.MODEL.DEVICE))
        quant_model = quantizer.quant_model.to(torch.device(cfg.MODEL.DEVICE))
    quant_model.eval()

    eval_quantize(cfg, quant_model)

    if args.quant_mode == 'calib':
        if args.fast_finetune:
            use_train = False
            quantizer.fast_finetune(eval_quantize, (cfg, quant_model, use_train))
            quant_model = quantizer.quant_model.to(torch.device(cfg.MODEL.DEVICE))
        quantizer.export_quant_config()
    if args.quant_mode == 'test' and args.dump_xmodel:
        if args.fast_finetune:
            quantizer.load_ft_param()
            quant_model = quantizer.quant_model.to(torch.device(cfg.MODEL.DEVICE))
        dump_xmodel(output_dir=cfg.OUTPUT_DIR, deploy_check=True)
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)