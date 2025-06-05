import math
from torch import nn
from models.nif.utils import exp_progression
from modules.quantizable.conv2d import QuantizableConv2d

class ConvolutionalMultiLayerPerceptron(nn.Module):
    def __init__(self, 
        input_features, 
        output_features, 
        architecture,
        activation,
        layer_params=dict(),
        residual=False
    ):
        super(ConvolutionalMultiLayerPerceptron, self).__init__()

        self.residual = residual

        hidden_sizes = exp_progression(**architecture)

        self.head = QuantizableConv2d(input_features, hidden_sizes[0], bias=True, **layer_params)

        body_layers = list()
        for i in range(0, len(hidden_sizes)-1):
            body_layers.append(QuantizableConv2d(hidden_sizes[i], hidden_sizes[i+1], bias=True, **layer_params))

        self.body = nn.Sequential(*body_layers)

        self.tail = QuantizableConv2d(hidden_sizes[-1], output_features, bias=True, **layer_params)

        nn.init.kaiming_uniform_(self.head.weight.get_raw_values(), a=math.sqrt(5))
        for layer in self.body:
            nn.init.kaiming_uniform_(layer.weight.get_raw_values(), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.tail.weight.get_raw_values(), a=math.sqrt(5))

        self.activation = activation

    def forward(self, x):
        y = x.movedim(-1, -3)

        y = self.head(y)
        y = self.activation(y)

        for i in range(0, len(self.body)):
            if self.residual:
                y = y + self.body[i](y)
            else:
                y = self.body[i](y)
            y = self.activation(y)

        y = self.tail(y)

        y = y.movedim(-3, -1)
        return y

