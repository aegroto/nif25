import torch
from torch.utils.tensorboard import SummaryWriter

class DummyWriter(object):
   def nop(*args, **kw): pass
   def __getattr__(self, _): return self.nop

WRITER = DummyWriter()
STEP = 0

def init_writer(id):
    global WRITER
    WRITER = SummaryWriter(log_dir = id)

def initialized():
    return WRITER is not None

def step_interval(interval=500):
    return initialized() and STEP % interval == 0

def signed_tensor(tensor, normalize=False):
    if normalize:
        normalized = tensor / tensor.abs().max()
    else:
        normalized = tensor

    negative = -(normalized * (tensor < 0.0).float())
    positive = normalized * (tensor > 0.0).float()

    return torch.cat([negative, positive, tensor.mul(0.0)], -1)

