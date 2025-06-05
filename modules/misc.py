import torch
class Constant(torch.nn.Module):
    def __init__(self, value=0.0):
        super(Constant, self).__init__()
        self.register_buffer("value", torch.Tensor([value]), persistent=False)

    def forward(self, _):
        return self.value