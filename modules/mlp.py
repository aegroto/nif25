import math
from torch import nn
from models.nif.utils import exp_progression
from modules.quantizable.classic_linear import QuantizableClassicLinear

class MultiLayerPerceptron(nn.Module):
    def __init__(self, 
        input_features, 
        output_features, 
        architecture,
        activation,
        final_activation=None,
        residual=False
    ):
        super(MultiLayerPerceptron, self).__init__()

        self.residual = residual

        hidden_sizes = exp_progression(**architecture)

        self.head = QuantizableClassicLinear(input_features, hidden_sizes[0])

        body_layers = list()
        for i in range(0, len(hidden_sizes)-1):
            body_layers.append(QuantizableClassicLinear(hidden_sizes[i], hidden_sizes[i+1]))

        self.body = nn.Sequential(*body_layers)

        self.tail = QuantizableClassicLinear(hidden_sizes[-1], output_features)

        nn.init.kaiming_uniform_(self.head.weight.get_raw_values(), a=math.sqrt(5))
        for layer in self.body:
            nn.init.kaiming_uniform_(layer.weight.get_raw_values(), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.tail.weight.get_raw_values(), a=math.sqrt(5))

        self.activation = activation
        self.final_activation = final_activation if final_activation is not None else nn.Identity()

    def forward(self, x):
        y = self.head(x)
        y = self.activation(y)

        for i in range(0, len(self.body)):
            if self.residual:
                y = y + self.body[i](y)
            else:
                y = self.body[i](y)
            y = self.activation(y)

        y = self.tail(y)
        y = self.final_activation(y)
        return y

