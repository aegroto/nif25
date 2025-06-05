import numpy
import torch

from models.nif.utils import exp_progression

class SpectraPositionalEncoder(torch.nn.Module):
    def __init__(self, in_features, spectra):
        super(SpectraPositionalEncoder, self).__init__()

        self.in_features = in_features

        periods = list()
        for spectrum_config in spectra:
            spectrum_periods = exp_progression(**spectrum_config, type = numpy.float32) 
            periods = periods + spectrum_periods

        self.register_buffer("periods", torch.tensor(periods), persistent=False)

        # print(f"Periods: {self.periods.tolist()}")

        self.out_dim = in_features + in_features * 2 * len(periods)

    def forward(self, x):
        angles = self.periods * torch.pi * x.unsqueeze(-1) 
        angles = angles.flatten(-2, -1)
        return torch.cat([
            x,
            torch.sin(angles),
            torch.cos(angles)
        ], -1)
