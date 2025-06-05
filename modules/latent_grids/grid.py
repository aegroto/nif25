import math
import torch
from torch import nn
from modules.latent_grids.features import GridFeatures
from modules.latent_grids.learned_upsampler import UpshuffleUpsampler


class Grid(nn.Module):
    def __init__(self, 
                reference_height,
                reference_width,
                upsampler_params=None,
                downscale=0,
                num_features=1,
                init_range=None):
        super().__init__()

        self.init_range = init_range

        if not upsampler_params:
            upsampler_params = dict()
        scale = int(2 ** downscale)

        self.upsampler = UpshuffleUpsampler(downscale, **upsampler_params)

        grid_height = int(reference_height // scale)
        grid_width = int(reference_width // scale)
        self.grid_features = GridFeatures(num_features, grid_height, grid_width, init_range)

        # self.grid_features = NeuralGridFeatures()
        self.out_dim = num_features

        grid = self.grid_features.features.get_raw_values()
        size = torch.tensor(grid.shape[-2:], device=grid.device) - 1.0
        self.register_buffer("size", size)

    def forward(self, x):
        grid = self.grid_features(x)
        values = self.upsampler(grid)
        return values
