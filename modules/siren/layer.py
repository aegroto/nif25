import numpy
import torch
from torch import nn

from models.nif.utils import exp_progression
from modules.quantizable.linear import QuantizableLinear
from modules.siren.activation import SelfModulatedSine
from modules.siren.utils import build_first_layer_initializer, build_linear_initializer

class SelfModulatedSirenLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_params,
        is_first=False,
        omega=30.0,
        c=6.0,
    ):
        super(SelfModulatedSirenLayer, self).__init__()

        self.linear = QuantizableLinear(in_features, out_features, biased=False)
        self.activation = SelfModulatedSine(omega, out_features, **activation_params)

        with torch.no_grad():
            if is_first:
                self.linear.apply(build_first_layer_initializer())
            else:
                self.linear.apply(build_linear_initializer(c, omega))

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        return y
