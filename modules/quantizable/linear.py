import debug
import numpy
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.nif.utils import exp_progression

from modules.quantizable import QuantizableParameter

class QuantizableLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        biased=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = QuantizableParameter(torch.empty((out_features, in_features)))

        if biased:
            self.bias = QuantizableParameter(torch.empty((out_features)))
        else:
            self.bias = None

    def forward(self, x):
        weight = self.weight.get()
        bias = self.bias.get() if self.bias is not None else None

        y = F.linear(x, weight, bias)

        if debug.step_interval():
            dump_weight = weight.abs() / weight.abs().max()
            dump_weight = dump_weight.unsqueeze(0).unsqueeze(0)
            debug.WRITER.add_image(f"weight.{self.in_features}x{self.out_features}", dump_weight, debug.STEP, dataformats="NCHW")

        return y

    def __repr__(self) -> str:
        return "QuantizableLinear(in_features={}, out_features={})".format(
            self.in_features, self.out_features
        )
