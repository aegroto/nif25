import debug
import math
import torch

from losses.chroma import init_chroma_weight

import torch.nn.functional as F

def log_cosh(x: torch.Tensor) -> torch.Tensor:
    return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)

class LogCoshLoss(torch.nn.Module):
    def __init__(self, chroma_weight=None):
        super().__init__()
        self.weight = init_chroma_weight(chroma_weight)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        diff = y_pred - y_true
        if self.weight:
            diff = self.weight(diff)

        return torch.mean(log_cosh(diff))

class DistributedLogCoshLoss(torch.nn.Module):
    def __init__(self, shuffle=1):
        super().__init__()
        self.shuffle = shuffle

    def __call__(self, original, prediction):
        diff = original - prediction
        abs_logcosh = log_cosh(diff)

        mean_logcosh = abs_logcosh.mean()

        abs_logcosh_blocks = F.pixel_unshuffle(abs_logcosh.unsqueeze(1), self.shuffle)
        mean_logcosh_blocks = abs_logcosh_blocks.mean(1)

        blocks_divergence = F.relu(mean_logcosh_blocks - mean_logcosh)

        if debug.step_interval():
            dump_divergence = blocks_divergence.abs().unsqueeze(0) / blocks_divergence.abs().max()
            dump_logcosh = abs_logcosh.unsqueeze(0)
            debug.WRITER.add_images("blocks_divergence", dump_divergence, debug.STEP, dataformats="CNHW")
            debug.WRITER.add_images("log_cosh", dump_logcosh, debug.STEP, dataformats="CNHW")

        return blocks_divergence.mean()

