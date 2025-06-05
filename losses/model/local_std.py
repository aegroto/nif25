import torch
import math

from losses.model import ModelLoss, QuantizerLoss

class LowLocalStdLoss(ModelLoss):
    def __init__(self, shuffle=2, regex=None, first_dim=-3):
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
            std = abs.std(1)
            local_max = abs.max(1, keepdim=True).values

            magnitudes = std / (1.0e-6 + local_max) # (1.0 + abs) / (1.0 + abs.max())
            total += magnitudes.abs().mean()

        return total 


class LowFeaturewiseStdLoss(ModelLoss):
    def __init__(self, regex=None, first_dim=1):
        super().__init__(regex)
        self.first_dim = first_dim

    def _call(self, parameters):
        total = torch.zeros((), device=parameters[0].device)

        for parameter in parameters:
            stds = (parameter / parameter.max()).std(1)
            total += stds.mean()

        return total 

class QuantizedStd(QuantizerLoss):
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

            p = values.flatten(-2, -1)
            # p = p.flatten(0, -2)

            unrounded = quantizer.quantize_unrounded(p)
            # std = unrounded.std() / quantizer.inner_quantizer.max_symbol

            std = unrounded.std(1)
            max = unrounded.abs().max(1).values

            norm_std = std / (1.0e-6 + max)

            total += norm_std.mean() * (size / self._max_size())

        return total / len(modules)
