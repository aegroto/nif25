import math
import torch
from torch import nn
from torch.nn import functional as F

from modules.quantizable import QuantizableParameter

class QuantizableConv2d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size=3,
        padding=1,
        bias=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.padding = padding
        self.kernel_size = kernel_size

        if bias:
            self.bias = QuantizableParameter(torch.zeros((out_features))) 
        else:
            self.bias = None

        self.weight = QuantizableParameter(
            torch.zeros((out_features, in_features, kernel_size, kernel_size)))

    def forward(self, x):
        weight = self.weight.get()
        if self.bias is not None:
            bias = self.bias.get()
        else:
            bias = None

        y = F.conv2d(x, weight, bias=bias, padding=self.padding)
        return y

    def __repr__(self) -> str:
        return "QuantizableConv2d({}, {}, {}x{})".format(
            self.in_features, self.out_features, self.kernel_size, self.kernel_size
        )
