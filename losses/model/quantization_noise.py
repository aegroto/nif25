import torch

from losses.model import QuantizerLoss

class QuantizationNoiseLoss(QuantizerLoss):
    def __init__(self, regex=None, amount=1.0):
        super().__init__(regex)
        self.amount = amount

    def _call(self, modules):
        total = torch.zeros((), device=modules[0]._values.device)
        for module in modules:
            quantizer = module.quantizer
            values = module._values

            unrounded = quantizer.quantize_unrounded(values)
            noise = torch.sin(torch.pi * unrounded).abs().mean()
            noise = torch.nn.functional.dropout(noise, 1.0 - self.amount)

            total += noise

        return total / len(modules)
