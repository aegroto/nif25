import copy
import math
import torch
from torch import nn
from compression.quantization import quantizer_from_config

class QuantizableParameter(nn.Module):
    def __init__(self, data, requires_grad=True):
        super().__init__()

        self.quantizer = None
        self.__amount = 0.0

        self._values = nn.Parameter(data, requires_grad=requires_grad)

    def initialize_quantizer(self, parameters):
        config = copy.deepcopy(parameters)
        config["target_tensor"] = self._values
        config["id"] = "_values"
        config["device"] = self._values.device
        self.quantizer = quantizer_from_config(config)

    def set_quantization_amount(self, value):
        self.__amount = value
        self.quantizer.get_rounder().set_amount(value)

    def set_quantization_bits(self, value):
        self.quantizer.set_bits(value)

    def __quantize_param(self, param):
        if self.quantizer is not None:
            quantization_result = self.quantizer.quantize_tensor(param)
            dequantized_param = self.quantizer.dequantize_tensor(quantization_result)
            return dequantized_param
        else:
            return param

    def get_raw_values(self):
        return self._values

    def apply_quantization(self):
        dequantized_values = self.__quantize_param(self._values)
        self._values.set_(dequantized_values)

    def get(self):
        if self.__amount == 0:
            values = self._values
        else:
            dequantized_values = self.__quantize_param(self._values)
            error = dequantized_values - self._values
            values = self._values + error.detach()
        
        return values
