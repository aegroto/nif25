import torchvision
import torch

from phases.infer import get_padding_for_shuffling

class ShufflePadder(torch.nn.Module):
    def __init__(self, height, width, shuffle) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.padding = get_padding_for_shuffling(height, width, shuffle)

    def forward(self, x):
        return torch.nn.functional.pad(x, self.padding, "replicate")

class ShuffleUnpadder(torch.nn.Module):
    def __init__(self, height, width, shuffle) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.padding = get_padding_for_shuffling(height, width, shuffle)

    def forward(self, x):
        padding = get_padding_for_shuffling(x.size(-3), x.size(-2). shuffle)
        height_padding = padding[3]
        width_padding = padding[1]

        unpadded_height = self.height - height_padding
        unpadded_width = self.width - width_padding

        unpadded = torchvision.transforms.functional.crop(
            x,
            0, 0,
            unpadded_height,  unpadded_width
        )

        return unpadded

