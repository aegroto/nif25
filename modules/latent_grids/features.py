import torch
from torch import nn
from modules.quantizable import QuantizableParameter
from modules.siren import Siren

class GridFeatures(nn.Module):
    def __init__(self, num_features, height, width, init_range=None):
        super().__init__()

        self.init_range = init_range

        grid_height = height
        grid_width = width

        self.features = QuantizableParameter(torch.zeros((num_features, grid_height, grid_width)))
        if init_range is not None:
            nn.init.uniform_(self.features.get_raw_values(), init_range[0], init_range[1])

    def forward(self, _):
        return self.features.get().movedim(0, -1)

    def extra_repr(self) -> str:
        return "{}".format(self.features.get_raw_values().shape)
