import math
import torch
from torch import nn
from modules.quantizable.conv2d import QuantizableConv2d

class ConvolutionalGridUpsampler(nn.Module):
    def __init__(self, 
                kernel_size=3,
                num_features=1,
                downshuffle=1, 
                upsample_factor=1, upsample_iterations=1):
        super().__init__()

        self.downshuffle = downshuffle
        self.upsample_per_iteration = upsample_factor
        self.upsample_iterations = upsample_iterations
        self.upsample = upsample_factor ** upsample_iterations

        upsampler_in = num_features * self.downshuffle ** 2
        upsampler_out = num_features * ((self.downshuffle * self.upsample_per_iteration) ** 2)
        self.transform = nn.Sequential(
            QuantizableConv2d(upsampler_in, upsampler_out, kernel_size=kernel_size, padding=kernel_size // 2) 
        )

        for layer in self.transform:
            if hasattr(layer, "weight"):
                nn.init.kaiming_uniform_(layer.weight.get_raw_values(), a=math.sqrt(5))
        
        # mean = 1.0 / upsampler_in
        # nn.init.uniform_(self.transform.weight.get_raw_values(), 
        #                  mean * (1.0 - upsample_init_noise), mean * (1.0 + upsample_init_noise))

    def __upsample(self, grid, iteration=1):
        grid = grid.movedim(-1, 0)
        grid = torch.nn.functional.pixel_unshuffle(grid, self.downshuffle)
        # grid = grid.movedim(0, -1)

        grid = self.transform(grid)

        # grid = grid.movedim(-1, 0)
        grid = torch.nn.functional.pixel_shuffle(grid, self.downshuffle * self.upsample_per_iteration)
        grid = grid.movedim(0, -1)
        return grid

    def forward(self, grid):
        for i in range(0, self.upsample_iterations):
            grid = self.__upsample(grid, i + 1)

        return grid

        

