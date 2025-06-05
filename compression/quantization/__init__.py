import copy
from compression.quantization.uniform import UniformQuantizer

def quantizer_from_config(parameters):
    config = copy.deepcopy(parameters)
    mode = config["mode"]
    del config["mode"]

    if mode is not None:
        if mode == "uniform":
            return UniformQuantizer(**config)

