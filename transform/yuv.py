import kornia
import torch

class RGBToYCbCr(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor):
        return kornia.color.ycbcr.rgb_to_ycbcr(tensor)

class YCbCrToRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor):
        return kornia.color.ycbcr.ycbcr_to_rgb(tensor)
