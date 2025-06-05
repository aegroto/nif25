import numpy
import torch
from torch import nn

from models.nif.utils import exp_progression

class SelfModulatedSine(nn.Module):
    def __init__(
        self,
        omega,
        features,
        period_modulations,
        phase_sampling_period
    ):
        super(SelfModulatedSine, self).__init__()

        self.omega = omega
            
        self.register_buffer(
            "phases", 
            torch.linspace(-phase_sampling_period, phase_sampling_period, features)
                .mul(2.0 * torch.pi)
                .sin()
                .mul(torch.pi),
            False
        )

        self.register_buffer(
            "periods", 
            torch.tensor(exp_progression(**period_modulations, type=numpy.float32, length=features)),
            False
        )

    def forward(self, x):
        omega = self.omega * self.periods
        phi = self.phases

        y = torch.sin(omega * x + phi)

        return y
