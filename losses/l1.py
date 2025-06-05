import debug
import torch

from losses.chroma import init_chroma_weight

import torch.nn.functional as F

class L1Loss(torch.nn.Module):
    def __init__(self, chroma_weight=None):
        super().__init__()
        self.weight = init_chroma_weight(chroma_weight)

    def __call__(self, img1, img2):
        diff = img1 - img2
        if self.weight:
            diff = self.weight(diff)
        return torch.mean(torch.abs(diff))

class DistributedL1Loss(torch.nn.Module):
    def __init__(self, shuffle=1):
        super().__init__()
        self.shuffle = shuffle

    def __call__(self, original, prediction):
        diff = original - prediction
        abs_diff = diff.abs()

        mean_diff = abs_diff.mean()

        abs_diff_blocks = F.pixel_unshuffle(abs_diff.unsqueeze(1), self.shuffle)
        mean_diff_blocks = abs_diff_blocks.mean(1)

        blocks_divergence = F.relu(mean_diff_blocks - mean_diff)

        if debug.step_interval():
            dump_divergence = blocks_divergence.abs().unsqueeze(0) / blocks_divergence.abs().max()
            dump_diff = diff.abs().unsqueeze(0)
            debug.WRITER.add_images("blocks_divergence", dump_divergence, debug.STEP, dataformats="CNHW")
            debug.WRITER.add_images("diff", dump_diff, debug.STEP, dataformats="CNHW")

        return blocks_divergence.mean()


