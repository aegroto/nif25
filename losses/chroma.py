import torch

from utils import load_device

def init_chroma_weight(params):
    if params:
        return LossChromaWeight(**params)
    else:
        return None

class LossChromaWeight(torch.nn.Module):
    def __init__(self, y_weight=0.2, cb_weight=0.4, cr_weight=0.4):
        super().__init__()
        self.weights = torch.Tensor([y_weight, cb_weight, cr_weight ]) \
            .unsqueeze(-1).unsqueeze(-1).to(load_device())

    def forward(self, y):
        return y * self.weights
