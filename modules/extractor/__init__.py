import torch
import math
from torch import nn
from models.nif.utils import exp_progression
from modules.quantizable.classic_linear import QuantizableClassicLinear
from modules.quantizable.conv2d import QuantizableConv2d

import torch.nn.functional as F

class DenoisingBlock(nn.Module):
    def __init__(self, 
        input_features, 
        hidden_features,
        shuffle
    ):
        super(DenoisingBlock, self).__init__()

        self.shuffle = shuffle
        
        block_features = input_features * (shuffle ** 2)
        self.pre = QuantizableClassicLinear(block_features, hidden_features)
        self.post = QuantizableClassicLinear(hidden_features, block_features)

        nn.init.kaiming_uniform_(self.pre.weight.get_raw_values(), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.post.weight.get_raw_values(), a=math.sqrt(5))

    def forward(self, x):
        y = x

        y = y.movedim(-1, 0)
        y = F.pixel_unshuffle(y, self.shuffle)
        y = y.movedim(0, -1)

        y = self.pre(y)
        y = F.gelu(y)
        y = self.post(y)

        y = y.movedim(-1, 0)
        y = F.pixel_shuffle(y, self.shuffle)
        y = y.movedim(0, -1)

        y = y - x

        return y

class Extractor(nn.Module):
    def __init__(self, 
        input_features, 
        output_features, 
        shuffle,
        architecture
    ):
        super(Extractor, self).__init__()

        hidden_sizes = exp_progression(**architecture)

        self.head = QuantizableClassicLinear(input_features, input_features)

        body_layers = list()
        for i in range(0, len(hidden_sizes)-1):
            body_layers.append(DenoisingBlock(input_features, hidden_sizes[i], shuffle))

        self.body = nn.Sequential(*body_layers)

        self.tail = QuantizableClassicLinear(input_features, output_features)

        nn.init.kaiming_uniform_(self.head.weight.get_raw_values(), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.tail.weight.get_raw_values(), a=math.sqrt(5))

    def _activate(self, y):
        return torch.nn.functional.gelu(y)

    def forward(self, x):
        y = self.head(x)
        y = F.gelu(y)

        for i in range(0, len(self.body)):
            y = self.body[i](y)


        y = self.tail(y)
        return y

