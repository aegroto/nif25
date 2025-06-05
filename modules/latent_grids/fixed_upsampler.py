import torchvision
import math
import torch
from torch import nn

class BilinearGridUpsampler(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, grid, target_size):
        grid = torch.nn.functional.interpolate(grid.movedim(-1, 0).unsqueeze(1), 
                                            scale_factor=self.scale,
                                            mode="bilinear")
        grid = grid.squeeze(1).movedim(0, -1)
        return grid

class GaussianGridUpsampler(nn.Module):
    def __init__(self,
                num_features=1,
                downshuffle=1, 
                upsample_factor=1, upsample_iterations=1):
        super().__init__()

        self.upsample = upsample_factor ** upsample_iterations

    def forward(self, grid):
        grid = torch.nn.functional.interpolate(grid.movedim(-1, 0).unsqueeze(0), 
                                               scale_factor=self.upsample,
                                               mode="nearest"
                                    )
        
        grid = torchvision.transforms.functional.gaussian_blur(grid, 7, 30.0)
        grid = grid.movedim(1, -1).squeeze(0)

        return grid
        


class UniformGridUpsampler(nn.Module):
    def __init__(self,
                kernel_size=7):
        super().__init__()

        self.kernel_size = kernel_size
        self.register_buffer("kernel", torch.zeros(1, 1, self.kernel_size, self.kernel_size), persistent=False)
        with torch.no_grad():
            self.kernel.fill_(1.0 / self.kernel.numel())

    def forward(self, grid, target_size):
        grid = torch.nn.functional.interpolate(grid.movedim(-1, 0).unsqueeze(1), 
                                               size=target_size,
                                               mode="nearest"
                                    )
        grid = torch.nn.functional.conv2d(grid, self.kernel, padding=self.kernel_size//2)
        grid = grid.squeeze(1).movedim(0, -1)

        return grid
        