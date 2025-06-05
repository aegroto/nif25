from torch import nn
from losses.charbonnier import CharbonnierLoss
from losses.composite import CompositeLoss
from losses.gradient import SobelLoss
from losses.l1 import DistributedL1Loss, L1Loss
from losses.l2 import DistributedL2Loss, L2Loss, RelativeL2Loss
from losses.log_cosh import DistributedLogCoshLoss, LogCoshLoss
from losses.lpips import LPIPSLoss
from losses.model.dispersion import HighLocalDispersionLoss
from losses.model.gaussian import LowGaussianResidualLoss
from losses.model.gradient import LowGradientLoss, QuantizedGradient
from losses.model.local_std import LowFeaturewiseStdLoss, LowLocalStdLoss, QuantizedStd
from losses.model.magnitude import LowLocalMagnitudeLoss, LowMagnitudeLoss, QuantizedMagnitude
from losses.model.quantization_noise import QuantizationNoiseLoss
from losses.ssim import DChannelWiseSSIMLoss, DMS_SSIMLoss, DSSIMLoss

def loss_from_id(id, model, args):
    if id == "l1":
        return L1Loss(**args)
    if id == "distributed_l1":
        return DistributedL1Loss(**args)
    if id == "l2":
        return L2Loss(**args)
    if id == "distributed_l2":
        return DistributedL2Loss(**args)
    if id == "log_cosh":
        return LogCoshLoss(**args)
    if id == "distributed_log_cosh":
        return DistributedLogCoshLoss(**args)
    if id == "ssim":
        return DSSIMLoss(model.generator.target_postprocessor, **args)
    if id == "channelwise_ssim":
        return DChannelWiseSSIMLoss(model.generator.target_postprocessor, **args)
    if id == "ms_ssim":
        return DMS_SSIMLoss(**args)
    if id == "sobel":
        return SobelLoss(**args)
    if id == "lpips":
        return LPIPSLoss(**args)
    if id == "charbonnier":
        return CharbonnierLoss(**args)

    if id == "low_magnitude":
        return LowMagnitudeLoss(**args)
    if id == "low_local_std":
        return LowLocalStdLoss(**args)
    if id == "low_featurewise_std":
        return LowFeaturewiseStdLoss(**args)
    if id == "high_dispersion":
        return HighLocalDispersionLoss(**args)
    if id == "quantized_magnitude":
        return QuantizedMagnitude(**args)
    if id == "quantized_std":
        return QuantizedStd(**args)
    if id == "gaussian":
        return LowGaussianResidualLoss(**args)
    if id == "quantization_noise":
        return QuantizationNoiseLoss(**args)
    if id == "low_local_magnitude":
        return LowLocalMagnitudeLoss(**args)
    if id == "low_gradient":
        return LowGradientLoss(**args)
    if id == "quantized_gradient":
        return QuantizedGradient(**args)

def build_loss_fn(components_config, model, return_components=False):
    losses = list()
    for loss_config in components_config:
        if "args" in loss_config:
            args = loss_config["args"]
        else:
            args = dict()

        if "weight" in loss_config:
            weight = loss_config["weight"]
        else:
            weight = 1.0

        losses.append((loss_from_id(loss_config["type"], model, args), weight))

    return CompositeLoss(losses, return_components)