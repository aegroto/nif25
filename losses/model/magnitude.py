import re
import torch
import statistics

from losses.log_cosh import log_cosh
from losses.model import ModelLoss, QuantizerLoss

class LowMagnitudeLoss(ModelLoss):
    def __init__(self, regex=None):
        super().__init__(regex)

    def _call(self, parameters):
        total = torch.zeros((), device=parameters[0].device)

        for parameter in parameters:
            abs = parameter.abs()
            magnitudes = abs / (1.0e-6 + abs.max()) # (1.0 + abs) / (1.0 + abs.max())
            total += magnitudes.abs().mean()

        return total / len(parameters)

class LowLocalMagnitudeLoss(ModelLoss):
    def __init__(self, regex=None, shuffle=2, first_dim=-3):
        super().__init__(regex)
        self.shuffle = shuffle
        self.first_dim = first_dim

    def _call(self, parameters):
        total = torch.zeros((), device=parameters[0].device)

        for parameter in parameters:
            p = parameter.flatten(0, self.first_dim)
            p = p.unsqueeze(1)
            p = torch.nn.functional.pixel_unshuffle(p, self.shuffle)

            abs = p.abs()
            local_max = abs.max(1, keepdim=True).values

            magnitudes = abs / (1.0e-6 + local_max) # (1.0 + abs) / (1.0 + abs.max())
            total += magnitudes.abs().mean()

        return total / len(parameters)

class QuantizedMagnitude(QuantizerLoss):
    def __init__(self, regex=None, recalibration_interval=1):
        super().__init__(regex)
        self.__recalibration_interval = recalibration_interval
        self.__current_interval = 0

    def _call(self, modules):
        total = torch.zeros((), device=modules[0]._values.device)
        for module in modules:
            quantizer = module.quantizer
            values = module._values

            self.__current_interval += 1
            if self.__current_interval == self.__recalibration_interval:
                quantizer.recalibrate(values)
                self.__current_interval = 0

            size = values.numel()

            unrounded = quantizer.quantize_unrounded(values)
            magnitude = unrounded.abs() / quantizer.max_symbol

            total += magnitude.mean() * (size / self._max_size())

        return total / len(modules)

