# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from . import losses
from .backbones import (
    BACKBONE_REGISTRY,
    build_resnet_backbone,
    build_backbone,
)
from .heads import (
    REID_HEADS_REGISTRY,
    build_heads,
    EmbeddingHead,
)
from .meta_arch import (
    build_model,
    META_ARCH_REGISTRY,
)

#* quant_model is used in QAT. QAT model in Vitis-AI2.5 has to abide following rules:
#* 1. replace + / torch.cat with pytorch_nndct.nn.modules.functional.Add / pytorch_nndct.nn.modules.functional.Cat
#* 2. If there are modules to be called multiple times(Usually these modules have no weights, like ReLU), 
#*    define multiple such modules and then call them separately in a forward pass.
#* 3. Insert QuantStub and DeQuantStub to define beginning and end of the nerwork which to be quantized.
from .quant_model import(
    build_quant_model,
    QUANT_MODEL_REGISTRY,
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]