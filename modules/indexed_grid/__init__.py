import numpy
import debug

import torch
from torch import nn
from models.nif.utils import exp_progression
from modules.mlp import MultiLayerPerceptron

from modules.quantizable import QuantizableParameter

class IndexedGrid(nn.Module):
    def __init__(self, 
                num_frequencies,
                scale,
                output_features,
                height=None,
                width=None,
                grid_scales=None):
        super(IndexedGrid, self).__init__()

        self.num_frequencies = num_frequencies
        self.scale = scale

        self.height = height
        self.width = width

        grid_scales = exp_progression(length=num_frequencies + 1, type=numpy.float32, **grid_scales)
        grid_features = list()

        for grid_scale in grid_scales:
            grid_features.append(QuantizableParameter(torch.zeros((output_features, 
                                              int(height * grid_scale),
                                              int(width* grid_scale),
                                            ))))

        self.grid_features = nn.ModuleList(grid_features)

        self.out_dim = output_features + 2 * num_frequencies * output_features

    def __index_in(self, c, features):
        h_coordinates = c[:,:,0]
        w_coordinates = c[:,:,1]

        h_indices = h_coordinates.mul(features.size(0) - 1).long()
        w_indices = w_coordinates.mul(features.size(1) - 1).long()

        f = features[h_indices, w_indices, :]
        return f

    def forward(self, c):
        in_c = list()
        in_g = list()
        out_f = list()

        for i in range(0, self.num_frequencies + 1):
            g = self.grid_features[i].get().movedim(0, -1)
            if debug.step_interval():
                with torch.no_grad():
                    in_g.append(torch.nn.functional.interpolate(
                        g.movedim(-1, 0).unsqueeze(1), size=(self.height, self.width)))

            if i == 0:
                coordinates = c
                in_c.append(coordinates)

                f0 = self.__index_in(coordinates, g)
                out_f.append(f0)
            else:
                coordinates = c.mul(torch.pi * self.scale ** i)
                sin_coordinates = coordinates.sin().abs()
                cos_coordinates = coordinates.cos().abs()
                in_c.append(sin_coordinates)
                in_c.append(cos_coordinates)

                f_sin = self.__index_in(sin_coordinates, g)
                f_cos = self.__index_in(cos_coordinates, g)
                out_f.append(f_sin)
                out_f.append(f_cos)

        f = torch.cat(out_f, -1)

        if debug.step_interval():
            with torch.no_grad():
                dump_g = torch.cat(in_g, 1).flatten(0, 1).unsqueeze(1)
                dump_g = dump_g.abs() / dump_g.abs().max() # .flatten(1, -1).max(1, keepdim=True).values
                debug.WRITER.add_images(f"grid_features", dump_g, debug.STEP, dataformats="NCHW")

                dump_c = torch.cat(in_c, -1).abs().movedim(-1, 0).unsqueeze(1)
                debug.WRITER.add_images(f"coordinates", dump_c, debug.STEP, dataformats="NCHW")

                dump_f = f.abs().movedim(-1, 0).unsqueeze(1) / f.abs().max() # .flatten(1, -1).max(1, keepdim=True).values
                debug.WRITER.add_images(f"output_grid_features", dump_f, debug.STEP, dataformats="NCHW")


        return f
