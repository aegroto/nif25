import torch
from torch import nn

from torchvision.transforms import Compose, ToTensor
from modules.positional_encoder import PositionalEncoder

from transform.normalize import RangeDenormalize, RangeNormalize
from transform.padding import ShufflePadder, ShuffleUnpadder
from transform.yuv import RGBToYCbCr, YCbCrToRGB

class NIFTargetPreProcessor(nn.Module):
    def __init__(self):
        super(NIFTargetPreProcessor, self).__init__()
        self.transform = Compose(
            [
                ToTensor(),
                RGBToYCbCr(),
                RangeNormalize(),
            ]
        )

    def forward(self, x):
        y = self.transform(x)
        return y

class NIFPostProcessor(nn.Module):
    def __init__(self):
        super(NIFPostProcessor, self).__init__()

        for t in ["mean"]:
            self.register_buffer(f"{t}_tensor", torch.zeros((3, 1, 1)), True)

        self.denormalize = RangeDenormalize()
        self.color_transform = YCbCrToRGB()

    def __override(self, name, value):
        getattr(self, name).set_(
            torch.tensor(value)
                .to(getattr(self, name).device)
                .reshape(getattr(self, name).shape)
        )

    def calibrate(self, image_tensor):
        means = [channel.mean() for channel in image_tensor.unbind(0)]
        self.__override("mean_tensor", means)
        target_tensor = image_tensor - self.mean_tensor

        return target_tensor

    def forward_no_color(self, x):
        y = x + self.mean_tensor
        return self.denormalize(y)

    def forward(self, x):
        y = self.forward_no_color(x)
        y = self.color_transform(y)
        return y
