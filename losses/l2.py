import debug
import torch

import torch.nn.functional as F

class L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, original, prediction):
        diff = original - prediction
        return diff.square().mean()

class RelativeL2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__epsilon = 1.0

    def __call__(self, original, prediction):
        diff = original - prediction
        return (diff.square() / (prediction.square() + self.__epsilon)).mean()

class DistributedL2Loss(torch.nn.Module):
    def __init__(self, shuffle=1):
        super().__init__()
        self.shuffle = shuffle

    def __call__(self, original, prediction):
        diff = original - prediction
        square_diff = diff.square()

        mean_diff = square_diff.mean()

        square_diff_blocks = F.pixel_unshuffle(square_diff.unsqueeze(1), self.shuffle)
        mean_square_diff_blocks = square_diff_blocks.mean(1)

        blocks_divergence = F.relu(mean_square_diff_blocks - mean_diff)

        if debug.step_interval():
            dump_divergence = blocks_divergence.abs().unsqueeze(0) / blocks_divergence.abs().max()
            dump_diff = diff.abs().unsqueeze(0)
            debug.WRITER.add_images("blocks_divergence", dump_divergence, debug.STEP, dataformats="CNHW")
            debug.WRITER.add_images("diff", dump_diff, debug.STEP, dataformats="CNHW")

        return blocks_divergence.mean()



        
