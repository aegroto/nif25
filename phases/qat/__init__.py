import torch
import copy
import re

def generate_quantization_config(config, model):
    quantization_config = dict()
    for name, module in model.named_modules():
        if not hasattr(module, "initialize_quantizer"):
            continue

        tensor_config = copy.deepcopy(config["quantization"]["default"])
        for group_config in config["quantization"]["groups"]:
            if re.search(group_config["regex"], name):
                tensor_config = copy.deepcopy(group_config)
                del tensor_config["regex"]

        tensor_config["rounding_mode"] = config["quantization"]["rounding_mode"]
        quantization_config[name] = tensor_config

    return quantization_config

def initialize_quantizers(quantization_config, model, prequantize_weights=False):
    for name, module in model.named_modules():
        if name not in quantization_config:
            continue

        tensor_config = quantization_config[name]
        module.initialize_quantizer(tensor_config)

        if prequantize_weights:
            with torch.no_grad():
                module.apply_quantization()

def set_quantization_amount(model, amount):
    for _, module in model.named_modules():
        if not hasattr(module, "set_quantization_amount"):
            continue

        module.set_quantization_amount(amount)

def recalibrate_quantizers(model):
    for _, module in model.named_modules():
        if not hasattr(module, "quantizer"):
            continue

        module.quantizer.recalibrate(module.get_raw_values())

def set_quantization_max_symbol(model, max_symbol):
    for _, module in model.named_modules():
        if not hasattr(module, "quantizer"):
            continue

        module.quantizer.set_max_symbol(max_symbol)

def apply_quantization(model):
    with torch.no_grad():
        for _, module in model.named_modules():
            if not hasattr(module, "apply_quantization"):
                continue
            module.apply_quantization()

