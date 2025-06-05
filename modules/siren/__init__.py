import torch
from torch import nn
from models.nif.utils import exp_progression
from modules.misc import Constant
from modules.quantizable.linear import QuantizableLinear

from modules.siren.layer import SelfModulatedSirenLayer
from modules.siren.utils import (
    Sine,
    build_first_layer_initializer,
    build_linear_initializer,
)


class Siren(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        architecture,
        activation_params=dict()
    ):
        super(Siren, self).__init__()

        hidden_sizes = exp_progression(**architecture)

        self.head = SelfModulatedSirenLayer(input_features, hidden_sizes[0], 
                                            activation_params, is_first=True)

        body_layers = list()
        for i in range(0, len(hidden_sizes) - 1):
            body_layers.append(SelfModulatedSirenLayer(hidden_sizes[i], hidden_sizes[i + 1], 
                                                       activation_params))

        self.body = nn.Sequential(*body_layers)

        self.tail = QuantizableLinear(hidden_sizes[-1], output_features)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y

