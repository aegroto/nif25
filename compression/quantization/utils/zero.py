import torch

def calculate_zero(tensor, config):
    mode = config["mode"]

    args = config["args"] if "args" in config else dict()

    if mode == "mean":
        return calculate_mean(tensor) 
    if mode == "mode":
        return calculate_mode(tensor) 
    elif mode == "local_mean":
        return calculate_local_mean(tensor, **args) 
    elif mode == "zero":
        return calculate_zero_zero(tensor, **args) 
    elif mode == "minimum":
        return calculate_minimum(tensor, **args) 
    else:
        return None

def calculate_mean(tensor):
    return tensor.mean()

def calculate_minimum(tensor):
    return tensor.min()

def calculate_mode(tensor):
    return tensor.flatten().mode().values

def calculate_local_mean(tensor, dim=0):
    return tensor.mean(dim, keepdim=True)

def calculate_zero_zero(tensor):
    return torch.zeros((1), device=tensor.device)
