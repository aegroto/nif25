# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from typing import List

from modules.quantizable import QuantizableParameter

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

class CoolChicUpsampling(nn.Module):
    def __init__(self, steps: int, upsampling_kernel_size: int = 2):
        """Instantiate an upsampling layer."""
        super().__init__()

        self.steps = steps

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
        # self.upsampling_layer = nn.ConvTranspose2d(1,1,upsampling_kernel.shape ,bias=False,stride=2)
        self.upsampling_kernel = QuantizableParameter(upsampling_kernel, requires_grad=False)


    def forward(self, grid: Tensor, target_size) -> Tensor:
        """From a list of C [1, C', H_i, W_i] tensors, where H_i = H / 2 ** i abd
            W_i = W / 2 ** i, upsample each tensor to H * W. Then return the values
            as a 4d tensor = [1, C' x C, H_i, W_i]

        Args:
            decoder_side_latent (List[Tensor]): a list of C latent variables
                with resolution [1, C', H_i, W_i].

        Returns:
            Tensor: The [1, C' x C, H_i, W_i] synthesis input.
        """

        # The main idea is to invert the batch dimension (always equal to 1 in our case)
        # with the channel dimension (not always equal to 1) so that the same convolution
        # is applied independently on the batch dimension.
        # upsampled_latent = rearrange(decoder_side_latent[-1], '1 c h w -> c 1 h w')

        print(grid.shape)
        upsampled = grid.movedim(-1, 0).unsqueeze(0)

        # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
        # target_tensor = rearrange(decoder_side_latent[i - 1], '1 c h w -> c 1 h w')

        for _ in range(0, self.steps):
            x_pad = F.pad(upsampled, self.upsampling_padding, mode='replicate')
            # upsampled = self.upsampling_layer(x_pad)
            upsampled = F.conv_transpose2d(x_pad, self.upsampling_kernel.get(), stride=2)

        # crop to remove padding in convolution
        H, W = upsampled.size()[-2:]
        upsampled = upsampled[
            :,
            :,
            self.upsampling_crop : H - self.upsampling_crop,
            self.upsampling_crop : W - self.upsampling_crop
        ]

        print(upsampled.shape)

        # crop to comply to higher resolution fm size before concatenation
        upsampled = upsampled[
            :,
            :,
            0 : target_size[0],
            0 : target_size[1]
        ]

        print(upsampled.shape)

        upsampled = upsampled.squeeze(0).movedim(0, -1)

        print(upsampled.shape)

        return upsampled
