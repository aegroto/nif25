import torch

from losses.model import ModelLoss

class HighLocalDispersionLoss(ModelLoss):
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
            # dispersion = p.abs().mean(1) / (1.0 + p.std(1))
            # dispersion = parameter.abs().mean() + 1.0 / (1.0 + parameter.std())
            dispersion = 1.0 / (1.0 + p.std(1))

            total += dispersion.mean()

        return total 
