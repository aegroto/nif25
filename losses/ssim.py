import torch
from pytorch_msssim import ssim, ms_ssim

from losses.chroma import init_chroma_weight

def reshape(tensor):
    shaped = tensor.unsqueeze(0).div(2.0).add(0.5)
    return shaped

def reshape_mono(tensor):
    return reshape(tensor.unsqueeze(0))

class DSSIMLoss:
    def __init__(self, postprocessor, chroma_weight=None):
        self.weight = init_chroma_weight(chroma_weight)
        self.postprocessor = postprocessor

    def __channel_ssim(self, ref, dist):
        return ssim(
            reshape_mono(ref), 
            reshape_mono(dist), 
            data_range=1.0
        ).reshape((1, 1, 1))

    def __call__(self, original, distorted):
        if self.weight:
            original = self.postprocessor.forward_no_color(original).unsqueeze(1)
            distorted = self.postprocessor.forward_no_color(distorted).unsqueeze(1)
            unweighted_ssim_val = ssim(original, distorted, data_range=1.0, size_average=False)           
            ssim_val = self.weight(unweighted_ssim_val).mean()
        else:
            ssim_val = ssim(
                self.postprocessor(original).unsqueeze(0), 
                self.postprocessor(distorted).unsqueeze(0), 
                data_range=1.0
            )

        return (1.0 - ssim_val)

class DChannelWiseSSIMLoss:
    def __init__(self, postprocessor):
        self.postprocessor = postprocessor

    def __call__(self, original, distorted):
        ssim_val = ssim(
            self.postprocessor(original).unsqueeze(1), 
            self.postprocessor(distorted).unsqueeze(1), 
            data_range=1.0
        )
        return (1.0 - ssim_val)

class DMS_SSIMLoss:
    def __init__(self):
        pass

    def __call__(self, original, distorted):
        ms_ssim_val = ms_ssim(
            original.unsqueeze(0), 
            distorted.unsqueeze(0), 
            data_range=1.0
        )
        return (1.0 - ms_ssim_val)
