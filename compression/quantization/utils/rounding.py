import math
import torch

def build_round_function(mode):
    if mode == "softround":
        return SoftRounder()
    if mode == "estimate":
        return EstimateRounder()
    if mode == "round":
        return HardRounder()
    else:
        return None

class SoftRounder():
    def __init__(self):
        self.set_amount(1.0)

    def set_amount(self, amount):
        self.alpha = max(20.0 * amount, 1.0)
        self.z = math.tanh(self.alpha / 2.) * 2.

    def round(self, x):
        m = torch.floor(x) + .5
        r = x - m
        y = m + torch.tanh(self.alpha * r) / self.z
        return y

class EstimateRounder():
    def __init__(self):
        self.max_exp_scale = (1.0 / 2.0)
        self.max_periodic_scale = (1.0 / 6.0)
        self.set_amount(1.0)

    def set_amount(self, amount):
        self.exp_scale = amount * self.max_exp_scale
        self.periodic_scale = amount * self.max_periodic_scale

    def round(self, x):
        two_pi_x = 2.0 * torch.pi * x

        exp = torch.exp(-self.exp_scale * torch.cos(two_pi_x))
        periodic = -self.periodic_scale * torch.sin(two_pi_x) 
        distance = periodic * exp 

        return (x + distance)

class HardRounder():
    def __init__(self):
        self.amount = 0.0

    def set_amount(self, amount):
        self.amount = amount

    def round(self, x):
        hard_rounded = torch.round(x)
        return hard_rounded

        # diff = hard_rounded - x
        # torch.nn.functional.dropout(diff, 1.0 - self.amount, inplace=True)
        # rounded = x + diff
        # return rounded

