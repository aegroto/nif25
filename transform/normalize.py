import torch

class RangeNormalize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.sub(0.5).mul(2.0)

class RangeDenormalize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.mul(0.5).add(0.5)