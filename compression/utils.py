import torch
import numpy

def numpy_type_for_bits(bits):
    if bits <= 8:
        return numpy.int8
    else:
        return numpy.int32

def torch_type_for_bits(bits):
    if bits <= 8:
        return torch.int8
    else:
        return torch.int32

