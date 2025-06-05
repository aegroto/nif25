import math
import torch
from torch import nn
import torch.nn.functional as F

from modules.quantizable.conv2d import QuantizableConv2d

class UpshuffleUpsampler(nn.Module):
    def __init__(self, steps, step_scale=2, kernel_size=3,
                    activation=None,
                    final_activation=None):
        super().__init__()
        self.steps = steps
        self.step_scale = step_scale

        self.activation = activation if activation is not None else nn.Identity()
        self.final_activation = final_activation if final_activation is not None else nn.Identity()

        self.padding = kernel_size // 2

        self.conv = QuantizableConv2d(1, self.step_scale ** 2, kernel_size=kernel_size, 
                                      padding=0) 

        mean = 1.0 / (kernel_size ** 2)
        torch.nn.init.uniform_(self.conv.weight.get_raw_values(), mean - 1.0e-4, mean + 1.0e-4)

    def forward(self, grid):
        grid = grid.movedim(-1, 0).unsqueeze(1)
        for i in range(0, self.steps):
            grid = F.pad(grid, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
            grid = self.conv(grid)
            grid = F.pixel_shuffle(grid, self.step_scale)
            if i < self.steps - 1:
                grid = self.activation(grid)
            
        grid = self.final_activation(grid)

        grid = grid.squeeze(1).movedim(0, -1)
        return grid


class ShuffleDownsampler(nn.Module):
    def __init__(self, steps, step_scale=2, kernel_size=3,
                    activation=None,
                    final_activation=None):
        super().__init__()
        self.steps = steps
        self.step_scale = step_scale

        self.activation = activation if activation is not None else nn.Identity()
        self.final_activation = final_activation if final_activation is not None else nn.Identity()

        self.padding = kernel_size // 2

        self.conv = QuantizableConv2d(self.step_scale ** 2, 1, kernel_size=kernel_size, 
                                      padding=0) 

        torch.nn.init.constant_(self.conv.weight.get_raw_values(), 1.0 / (kernel_size ** 2))

    def forward(self, grid):
        grid = grid.movedim(-1, 0).unsqueeze(1)
        for i in range(0, self.steps):
            grid = F.pixel_unshuffle(grid, self.step_scale)
            grid = F.pad(grid, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
            grid = self.conv(grid)
            if i < self.steps - 1:
                grid = self.activation(grid)
            
        grid = self.final_activation(grid)

        grid = grid.squeeze(1).movedim(0, -1)
        return grid

