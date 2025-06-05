import torch
import lpips

from utils import load_device

class LPIPSLoss(torch.nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        self.lpips.to(load_device())

    def __call__(self, img1, img2):
        return self.lpips(img1, img2).squeeze()
