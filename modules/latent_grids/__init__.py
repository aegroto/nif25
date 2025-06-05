import numpy
import debug

import torch
from torch import nn
from models.nif.utils import exp_progression

from modules.latent_grids.grid import Grid

class LatentGrids(nn.Module):
    def __init__(self, image_height, image_width, grids_params):
        super().__init__()

        grids = list()

        num_grids = grids_params["num_grids"]
        # num_features = exp_progression(length=num_grids, **grids_params["num_features"])
        # downscale = exp_progression(length=num_grids, **grids_params["downscale"])

        num_features = grids_params["num_features"]
        downscale = grids_params["downscale"]

        for i in range(0, num_grids):
            grids.append(Grid(image_height, image_width, 
                              num_features=num_features[i],
                              downscale=downscale[i],
                              **grids_params["extra"]))

        self.grids = nn.ModuleList(grids)

        self.out_dim = sum([grid.out_dim for grid in self.grids])

    def forward(self, x):
        y = torch.cat([grid(x) for grid in self.grids], -1)

        if debug.step_interval():
            for i in range(0, len(self.grids)):
                dump_y = self.grids[i].grid_features.features.get_raw_values()
                v = dump_y.abs().unsqueeze(0)
                m = dump_y.abs().flatten(-2, -1).max(-1).values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                dump_y = v / m 
                debug.WRITER.add_images(f"grid_features.{i}", dump_y, debug.STEP, dataformats="CNHW")

            v = y.abs().unsqueeze(0)
            m = y.abs().flatten(-2, -1).max(-1).values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            dump_upsampled_y = v / m
            debug.WRITER.add_images("upsampled_grid_features", dump_upsampled_y, debug.STEP, dataformats="CHWN")

        return y
