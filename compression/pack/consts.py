import torch

NO_QUANT_CONFIG = {
    "max_symbol": torch.tensor([0]),
    "bound": torch.zeros(1),
    "scale": torch.zeros(1),
    "zero": torch.zeros(1),
}

def is_no_quant(config):
    return config["max_symbol"] == 0
