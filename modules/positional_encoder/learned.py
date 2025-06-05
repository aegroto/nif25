import debug
import torch

from modules.quantizable import QuantizableParameter

class LearnedPositionalEncoder(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 num_frequencies,
                 min_period,
                 max_period):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        self.register_buffer("base_periods", torch.linspace(min_period, max_period, num_frequencies))

        self.amplitudes = QuantizableParameter(torch.ones(1, num_frequencies))
        # torch.nn.init.uniform_(self.amplitudes.get_raw_values(), 0.0, 1.0)

        self.out_dim = in_features + in_features * 2 * self.num_frequencies

    def forward(self, x):
        amplitudes = self.amplitudes.get()
        # amplitudes = torch.exp(10 * amplitudes - 4.0)

        periods = self.base_periods * amplitudes 

        if debug.step_interval():
            debug.WRITER.add_text("amplitudes", f"{amplitudes}", debug.STEP)
            debug.WRITER.add_text("periods", f"{periods}", debug.STEP)

        angles = periods * torch.pi * x.unsqueeze(-1) 

        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        return torch.cat([
            x,
            sin_angles.flatten(-2, -1),
            cos_angles.flatten(-2, -1),
        ], -1)

class NoisePositionalEncoder(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 start_frequency=0,
                 num_frequencies=None, 
                 scale=1.4):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.num_frequencies = num_frequencies

        self.register_buffer("periods", torch.Tensor([scale ** (start_frequency + i) for i in range(0, num_frequencies)]),
                             persistent=False)

        self.out_dim = in_features * 2 * self.num_frequencies

    def forward(self, x, weights=None):
        angles = self.periods * torch.pi * x.unsqueeze(-1) 
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        if weights is not None:
            sin_angles = sin_angles * weights.unsqueeze(-2)
            cos_angles = cos_angles * weights.unsqueeze(-2)
        return torch.cat([
            sin_angles.flatten(-2, -1),
            cos_angles.flatten(-2, -1),
        ], -1)

class SamplingPositionalEncoder(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 num_frequencies=None, 
                 scale=1.4):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.num_frequencies = num_frequencies

        self.register_buffer("periods", torch.Tensor([scale ** i for i in range(0, num_frequencies)]),
                             persistent=False)

        self.out_dim = in_features * self.num_frequencies

    def forward(self, x):
        angles = self.periods * torch.pi * x.unsqueeze(-1) 
        return torch.sin(angles)
