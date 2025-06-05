import torch

def itemize(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value
