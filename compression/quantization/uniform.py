import torch
import math

from compression.pack import UniformQuantizedPacker
from compression.quantization.utils.bound import calculate_bound
from compression.quantization.utils.convert import itemize
from compression.quantization.utils.rounding import build_round_function
from compression.quantization.utils.zero import calculate_zero

def calculate_scale(target_tensor, scale_factor=1.0):
    # b = (target_tensor.std() / target_tensor.abs().mean()) / math.sqrt(2.0)
    b = target_tensor.std() / math.sqrt(2.0)
    b *= scale_factor

    return 1.0 + b

class UniformQuantizer():
    def __init__(self, max_symbol=127, 
                 target_tensor=None, id=None, 
                 zero=None, zero_calculation=None,
                 bound=None, bound_calculation=None,
                 scale=None,
                 device=None,
                 rounding_mode="round"):
        self.id = id

        self.rounder = build_round_function(rounding_mode)

        self.rounding_mode = rounding_mode
        self.bound_calculation = bound_calculation
        self.zero_calculation = zero_calculation

        if target_tensor is not None:
            self.recalibrate(target_tensor)

        if zero is not None:
            self.zero = zero

        if bound is not None:
            self.bound = bound

        if scale is not None:
            self.scale = scale

        if device is not None:
            self.device = device

        self.max_symbol = max_symbol

        self.packer = UniformQuantizedPacker(self.max_symbol, self.device)

    def get_rounder(self):
        return self.rounder

    def set_max_symbol(self, max_symbol):
        self.max_symbol = max_symbol
    
    def recalibrate(self, target_tensor):
        self.zero = calculate_zero(target_tensor, self.zero_calculation)
        target_tensor = target_tensor - self.zero

        self.bound = calculate_bound(target_tensor, self.bound_calculation)
        target_tensor = target_tensor / self.bound

        self.scale = calculate_scale(target_tensor)

        self.device = target_tensor.device

    def __repr__(self):
        return f"({self.max_symbol}, {self.bound})"

    def get_config(self):
        return {
            "mode": "uniform",
            "max_symbol": itemize(self.max_symbol),
            "bound": itemize(self.bound),
            "scale": itemize(self.scale),
            "zero": itemize(self.zero),
            "rounding_mode": self.rounding_mode
        }

    def quantize_unrounded(self, tensor):
        tensor = tensor.sub(self.zero)

        tensor = tensor.clamp(-self.bound, self.bound)
        tensor = tensor.div(self.bound)
        tensor = tensor.sign() * tensor.abs().pow(1.0 / self.scale)
        tensor = tensor.mul(self.max_symbol)
        
        return tensor

    def quantize_tensor(self, tensor):
        return self.rounder.round(self.quantize_unrounded(tensor))

    def dequantize_tensor(self, tensor):
        tensor = tensor.div(self.max_symbol)
        tensor = tensor.sign() * tensor.abs().pow(self.scale)
        tensor = tensor.mul(self.bound)
        tensor = tensor.add(self.zero)
        return tensor
