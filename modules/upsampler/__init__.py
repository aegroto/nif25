# Deterministic upsampler, borrowed from https://github.com/Orange-OpenSource/Cool-Chic/blob/main/src/models/upsampling.py
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List

kernel_bicubic_alignfalse=[
    [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
    [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
    [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0308990479 ,-0.0926971436 ,0.2300262451 ,0.7724761963 ,0.7724761963 ,0.2300262451 ,-0.0926971436 ,-0.0308990479],
    [-0.0092010498 ,-0.0276031494 ,0.0684967041 ,0.2300262451 ,0.2300262451 ,0.0684967041 ,-0.0276031494 ,-0.0092010498],
    [0.0037078857 ,0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 ,0.0111236572 ,0.0037078857],
    [0.0012359619 ,0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 ,0.0037078857 ,0.0012359619],
]

class Upsampling(nn.Module):
    def __init__(self, upsampling_kernel_size: int):
        """Instantiate an upsampling layer.

        Args:
        """
        super().__init__()

        # adapted filter size
        K=upsampling_kernel_size
        self.upsampling_padding = (K//2, K//2, K//2, K//2)
        self.upsampling_crop=(3*K-2)//2

        # pad bicubic filter according to desired optimised kernel size
        K_bicubic=8 # default bicubic upsampling filter
        tmpad=(K-K_bicubic)//2
        kernel_pad=(tmpad, tmpad, tmpad, tmpad)
        upsampling_kernel=F.pad(torch.tensor(kernel_bicubic_alignfalse), kernel_pad) # padded with zeros by default

        # compatible with transpose conv
        upsampling_kernel=torch.unsqueeze(upsampling_kernel,0)
        upsampling_kernel=torch.unsqueeze(upsampling_kernel,0)

        # initialise layer and weights
        self.upsampling_layer = nn.ConvTranspose2d(1,1,upsampling_kernel.shape ,bias=False,stride=2)
        self.upsampling_layer.weight.data = nn.Parameter(upsampling_kernel,requires_grad=True)


    def forward(self, x: Tensor) -> Tensor:
        # upsampling kernel size, convolution padding and cropping preparation

        x_pad = F.pad(x, self.upsampling_padding,mode='replicate')
        y_conv= self.upsampling_layer(x_pad)

        # crop to remove padding in convolution
        H=y_conv.shape[-2]
        W=y_conv.shape[-1]
        upsampled_latent = y_conv[:, :, self.upsampling_crop:H-self.upsampling_crop, self.upsampling_crop:W-self.upsampling_crop]

        # crop to comply to higher resolution fm size before concatenation
        target_tensor = x
        upsampled_latent = upsampled_latent[:, :, 0:target_tensor.shape[-2], 0:target_tensor.shape[-1]]
        upsampled_latent = torch.cat((target_tensor, upsampled_latent), dim=0)

        # Here we return all the latent variables
        return upsampled_latent.permute((1,0,2,3))
