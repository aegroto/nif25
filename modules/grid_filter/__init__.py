import math
import debug
import torch
import torch.nn.functional as F

from torch import nn

from modules.quantizable import QuantizableParameter
from modules.siren.utils import Sine

def build_activation(name):
    if name == "gelu":
        return nn.GELU()

    if name == "relu":
        return nn.ReLU()

    if name == "sine" or name == "sine_first":
        return Sine()

    return nn.Identity()

def init_for_activation(weight, in_features, activation, omega=30.0):
    if activation == "sine_first":
        w_std = (1.0 / in_features)
        nn.init.uniform_(weight, -w_std, w_std)
    elif activation == "sine":
        w_std = (math.sqrt(6.0 / in_features) / omega)
        nn.init.uniform_(weight, -w_std, w_std)
    elif activation == "ones":
        nn.init.constant_(weight, 1.0)
    elif activation == "zeros":
        nn.init.constant_(weight, 0.0)
    else:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

class GridFilter(nn.Module):
    def __init__(self,
                 input_features,
                 height, width,
                 grid_size=None,
                 grid_scale=None,
                 activation=None,
                 force_init=None,
                 output_features=None,
                 residual=False,
                 shuffle=1,
                 biased=True,
                 upsampling_mode="nearest",
                 idx=None):
        super(GridFilter, self).__init__()

        self.idx = idx
        self.activation = build_activation(activation)

        self.upsampling_mode = upsampling_mode

        if residual:
            output_features = input_features
            shuffle = 1

        self.residual = residual
        self.shuffle = shuffle

        self.height = height
        self.width = width

        in_weight = input_features
        out_weight = output_features * (shuffle ** 2)

        if grid_scale:
            grid_size = (int(height * grid_scale), int(width * grid_scale))

        self.grid_weight = QuantizableParameter(torch.zeros((out_weight, in_weight, grid_size[0], grid_size[1])))

        weight = self.grid_weight.get_raw_values()
        init_for_activation(weight, in_weight, force_init or activation)

        if biased:
            self.grid_bias = QuantizableParameter(torch.zeros((out_weight, in_weight, grid_size[0], grid_size[1])))
            # nn.init.uniform_(self.grid_bias.get_raw_values(), -1.0e-4, 1.0e-4)
        else:
            self.grid_bias = None

        self.out_dim = input_features * output_features

    def forward(self, x):
        weight = self.grid_weight.get()
        bias = self.grid_bias.get() if self.grid_bias is not None else None

        if debug.step_interval():
            dump_weight = weight.abs() / weight.abs().max()
            dump_weight = dump_weight.flatten(0, 1).unsqueeze(1)
            debug.WRITER.add_images(f"filter.{self.idx}.weight", dump_weight, debug.STEP, dataformats="NCHW")

            if bias is not None:
                dump_bias = bias.abs() / bias.abs().max()
                dump_bias = dump_bias.flatten(0, 1).unsqueeze(1)
                debug.WRITER.add_images(f"filter.{self.idx}.bias", dump_bias, debug.STEP, dataformats="NCHW")

        weight = torch.nn.functional.interpolate(weight, size=(self.height, self.width), mode=self.upsampling_mode)
        if bias is not None:
            bias = torch.nn.functional.interpolate(bias, size=(self.height, self.width), mode=self.upsampling_mode)

        y = weight * x
        if bias is not None:
            y = y + bias

        y = y.flatten(0, 1)

        y = self.activation(y)

        if self.residual:
            y = x + y

        y = y.squeeze(0)

        return y

    def __repr__(self):
        return f"GridFilter({self.grid_weight.get_raw_values().shape}, {self.activation})"
