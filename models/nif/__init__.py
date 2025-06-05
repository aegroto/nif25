from torch import nn
from models.nif.generator import NIFGenerator

from modules.siren import Siren

class NIF(nn.Module):
    def __init__(
        self, 
        generator_params, 
        genesis_params,
        device=None
    ):
        super(NIF, self).__init__()
        self.generator = NIFGenerator(device=device, **generator_params)

        self.genesis = Siren(self.generator.out_dim, 3, **genesis_params)

    def forward(self, input):
        p = input
        y = self.genesis(p)
        y = y.movedim(-1, 0)
        return y
