import sys

import torch
import numpy
from torch import nn

def exp_progression(
        range, 
        e=1.0, 
        length=4,
        type=numpy.int32
    ):
    max_size = range[0] # max(range)
    min_size = range[1] # min(range) 
    factors = numpy.linspace(1.0, 0.0, num=length) ** e
    sizes = (max_size - min_size) * factors + min_size
    return sizes.astype(type).tolist()

def calculate_mean_tensor(image_tensor):
    values = list()
    for channel in image_tensor.unbind(0):
        values.append(channel.mean())

    return torch.tensor(values)

def load_image_data_into_params(params, height, width):
    params["generator_params"]["height"] = height
    params["generator_params"]["width"] = width
    return params

