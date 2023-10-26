# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from playreid.utils.registry import Registry


QUANT_MODEL_REGISTRY = Registry("QUANT")


def build_quant_model(cfg):
    quant_model = cfg.MODEL.META_ARCHITECTURE
    model = QUANT_MODEL_REGISTRY.get(quant_model)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
