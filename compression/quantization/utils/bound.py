def calculate_bound(tensor, config):
    mode = config["mode"]

    args = config["args"] if "args" in config else dict()

    if mode == "global":
        return calculate_global_bound(tensor) 
    elif mode == "local":
        return calculate_local_bound(tensor, **args) 
    elif mode == "quantile":
        return calculate_quantile_bound(tensor, **args) 
    else:
        return None

def calculate_global_bound(tensor):
    bound = tensor.abs().max()
    if bound == 0.0:
        bound = 1.0e-16
    return bound

def calculate_quantile_bound(tensor, quantile=1.0):
    return tensor.abs().quantile(quantile, interpolation="lower")

def calculate_local_bound(tensor, dim=0):
    t = tensor.flatten(dim+1, -1).flatten(0, dim)
    bound = t.abs().max(-1, keepdim=True).values
    bound[bound == 0.0] = 1.0e-16

    bound = bound.unsqueeze(-1).unsqueeze(-1)
    return bound
