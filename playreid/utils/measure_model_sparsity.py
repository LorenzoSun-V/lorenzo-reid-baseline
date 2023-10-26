# encoding: utf-8
"""
@author:  lorenzo
@contact: baiyingpoi123@gmail.com
"""
import torch
import logging
from thop import profile


logger = logging.getLogger(__name__)


def measure_module_sparsity(module, module_name, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
    sparsity = num_zeros / num_elements
    logger.info("layer_name:{}, num_zeros:{}, num_elements:{}, sparsity:{:.2f}".format(module_name, num_zeros, num_elements, sparsity))

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):
    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, module_name, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, module_name, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    logger.info("num_zeros:{}, num_elements:{}".format(num_zeros, num_elements))
    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def cal_sparsity(model):
    
    num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=False,
            linear_use_mask=False)
    logger.info("Global Sparsity: {:.2f}".format(sparsity))
    return sparsity


def get_parameter_number(model, device):
    dummy_input = torch.randn(1, 3, 256, 128).to(torch.device(device))
    flops, params = profile(model, (dummy_input,))
    logger.info(f'flops: {flops}, params: {params}')
    logger.info('flops: {:2f} M, params: {:2f} M'.format(flops / 1000000.0, params / 1000000.0))

