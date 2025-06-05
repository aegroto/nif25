# Uniform
# Bound: 0.06330465525388718
# Mean: 8.866067946655676e-05
# Maximum: 0.06321599334478378
# Average error: 0.0001234648225363344 (0.006819156464189291)
# Max error: 0.000249191332841292

# Square
# Bound: 0.06321599334478378
# Mean: tensor([0.], device='cuda:0')
# Maximum: 0.06321599334478378
# Average error: 7.060710777295753e-05 (0.006819156464189291)
# Max error: 0.0004499964416027069

import sys
import math
import torch
import numpy
from compression.quantization import quantizer_from_config
from compression.quantization.uniform import UniformQuantizer

torch.set_printoptions(sci_mode=False, linewidth=1000)
# tensor = torch.Tensor([1.0, 0.01, -1.0, 0.8, 0.6]).cuda()
tensor = torch.linspace(-1.0, 1.0, 8).pow(1.0).cuda()
# tensor = torch.zeros(8).cuda()
# tensor = tensor + torch.rand(tensor.shape, device=tensor.device) * (tensor.abs().mean() * 0.01)
# tensor = tensor.sort().values
# state_dict = torch.load("results/nif/test/fitted/state.pth")
# full_tensor = state_dict["genesis.body.2.weight._values"]
# tensor = full_tensor # [0,:8]

bits = 16

bound_calculation = { "mode": "global" }
zero_calculation = { "mode": "zero" }
quantizer = UniformQuantizer(bits=bits, target_tensor=tensor, 
                             bound_calculation=bound_calculation, zero_calculation=zero_calculation,
                             rounding_mode="round")

quantizer.get_rounder().set_amount(1.0)

# print(f"Bound: {quantizer.bound}")
# print(f"Mean: {quantizer.mean}")
print(f"Maximum: {tensor.abs().max()}")

# q = 0.99
# quantile = tensor.abs().quantile(q)
# print(f"Greater than {q}-th quantile {quantile}: {(tensor.abs() > quantile).count_nonzero()}")
# print(f"Greater than bound ({quantizer.bound}): {(tensor.abs() > quantizer.bound).count_nonzero()}")

result = quantizer.quantize_tensor(tensor)
dequantized = quantizer.dequantize_tensor(result)

noise = dequantized - tensor

if True:
    errors = (dequantized - tensor).abs()
    print(f"Average error: {errors.mean()} ({tensor.abs().mean()})")
    print(f"Max error: {errors.flatten().max()}")
    
if True:
    print("# Tensor")
    print(tensor)
    print("# Result")
    print(result)
    print("# Dequantized")
    print(dequantized)
    print("# Error")
    print(tensor - dequantized)
