import torchvision
import torch

from losses.model import ModelLoss

class LowGaussianResidualLoss(ModelLoss):
    def __init__(self, regex=None, kernel_size = 7, sigma = 10.0, first_dim=-3):
        super().__init__(regex)

        self.first_dim = first_dim

        self.kernel_size = kernel_size
        self.sigma = sigma

    def _call(self, parameters):
        total = torch.zeros((), device=parameters[0].device)

        for parameter in parameters:
            p = parameter.flatten(0, self.first_dim)
            original = p.unsqueeze(1) / (1.0e-6 + parameter.abs().max())
            blurred = torchvision.transforms.functional.gaussian_blur(original, self.kernel_size, self.sigma)

            residual = original - blurred
            total += residual.abs().mean() / (1.0e-6 + original.abs().mean())

        return total 
