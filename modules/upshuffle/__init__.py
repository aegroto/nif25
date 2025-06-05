import math
from torch import nn

import torch.nn.functional as F
from modules.grid_filter import build_activation, init_for_activation
from modules.quantizable.conv2d import QuantizableConv2d
from utils import get_or

class UpshufflingBlock(nn.Module):
    def __init__(self, input_features, 
                 shuffle=1,
                 output_features=None, 
                 residual=False,
                 channelwise=False,
                 kernel_size=1,
                 activation=None,
                 force_init=None,
        ):
        super(UpshufflingBlock, self).__init__()

        self.activation = build_activation(activation)
        self.residual = residual
        self.channelwise = channelwise

        in_conv = input_features
        if channelwise:
            in_conv = 1

        if residual:
            output_features = in_conv
            shuffle = 1

        self.shuffle = shuffle
        self.conv = QuantizableConv2d(
            in_conv, 
            output_features * (shuffle ** 2),
            kernel_size,
            padding = kernel_size // 2,
            bias = True
        )

        init_for_activation(self.conv.weight.get_raw_values(), input_features, get_or(force_init, activation))

        if channelwise:
            self.out_dim = input_features * output_features
        else:
            self.out_dim = output_features

    def forward(self, x):
        if self.channelwise:
            x = x.unsqueeze(1)

        y = self.conv(x)
        y = self.activation(y)
        y = F.pixel_shuffle(y, self.shuffle)
        if self.residual:
            y = x + y

        if self.channelwise:
            y = y.squeeze(1)
        return y
