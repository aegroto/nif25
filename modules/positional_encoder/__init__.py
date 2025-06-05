import torch

class PositionalEncoder(torch.nn.Module):
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

        self.out_dim = in_features + in_features * 2 * self.num_frequencies

    def forward(self, x):
        angles = self.periods * torch.pi * x.unsqueeze(-1) 
        angles = angles.flatten(-2, -1)
        return torch.cat([
            x,
            torch.sin(angles),
            torch.cos(angles)
        ], -1)
