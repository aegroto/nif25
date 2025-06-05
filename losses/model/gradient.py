import debug
import kornia
import torch

from losses.model import ModelLoss, QuantizerLoss

class LowGradientLoss(ModelLoss):
    def __init__(self, regex=None, first_dim=-3):
        super().__init__(regex)
        self.first_dim = first_dim

    def _call(self, parameters):
        total = torch.zeros((), device=parameters[0].device)

        for parameter in parameters:
            p = parameter.flatten(0, self.first_dim)
            p = p.unsqueeze(1)

            abs = p.abs()
            gradient = kornia.filters.spatial_gradient(abs, order=1)
            total += gradient.abs().mean() / (1.0e-6 + abs.max())

        return total / len(parameters)

class QuantizedGradient(QuantizerLoss):
    def __init__(self, regex=None, recalibration_interval=1):
        super().__init__(regex)
        self.__recalibration_interval = recalibration_interval
        self.__current_interval = 0

    def _call(self, modules):
        total = torch.zeros((), device=modules[0]._values.device)
        for (idx, module) in enumerate(modules):
            quantizer = module.quantizer
            values = module._values

            self.__current_interval += 1
            if self.__current_interval == self.__recalibration_interval:
                quantizer.recalibrate(values)
                self.__current_interval = 0

            p = module._values.flatten(0, -3)
            p = p.unsqueeze(-3)

            unrounded = quantizer.quantize_unrounded(p)

            abs = unrounded.abs()
            gradient = kornia.filters.spatial_gradient(abs, order=1, normalized=True)
            
            mean = gradient.abs().mean(-3).mean(0, keepdim=True)
            max = abs.max(0, keepdim=True).values

            norm_mean = mean / (1.0e-6 + max)

            total += norm_mean.mean()

            if debug.step_interval():
                debug.WRITER.add_images(f"gradient.{idx}", gradient.movedim(-3, -4).flatten(0, 1), debug.STEP, dataformats="NCHW")

        return total / len(modules)
