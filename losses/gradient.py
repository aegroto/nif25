import kornia
import torch

from losses.log_cosh import log_cosh

def grad(image):
    return kornia.filters.spatial_gradient(image.unsqueeze(0), order=1)

class SobelLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, reconstructed, original):
        diff = grad(reconstructed) - grad(original) 
        return torch.mean(log_cosh(diff))
