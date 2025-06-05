import math
import torch
from torch import nn
from torch.nn import functional as F

from modules.quantizable import QuantizableParameter

class QuantizableClassicLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = QuantizableParameter(torch.zeros((out_features, in_features)))
        self.bias = QuantizableParameter(torch.zeros((out_features)))

    def forward(self, x):
        y = F.linear(x, self.weight.get(), self.bias.get())
        return y

    def __repr__(self) -> str:
        return "QuantizableClassicLinear(in_features={}, out_features={})".format(
            self.in_features, self.out_features
        )
