import torch
import copy

from pytorch_msssim import ms_ssim
from infer import patched_infer
from losses.psnr import psnr


def eval_ms_ssim(context, reconstructed_image):
    return ms_ssim(
        reconstructed_image.unsqueeze(0), context.image.unsqueeze(0), data_range=1.0
    )


def eval_psnr(context, reconstructed_image):
    return psnr(reconstructed_image, context.image, 1.0)


def eval_loss(context, _):
    return context.avg_loss

def eval_weighted_distances(context, _):
    weights = context.config["selector"]["config"]["weights"]
    caps = context.config["selector"]["config"]["caps"]

    distances = [max(result.distance or result.value, torch.zeros(1, device=result.value.device)) for result in context.results]
    for i in range(0, len(caps)):
        if caps[i] is not None and distances[i] > caps[i]:
            return None

    value = sum([distance * weight for (distance, weight) in zip(weights, distances)])
    return value

def eval_func_from_id(id):
    if id == "psnr":
        return (eval_psnr, True)
    elif id == "ms_ssim":
        return (eval_ms_ssim, True)
    elif id == "loss":
        return (eval_loss, False)
    elif id == "weighted_distances":
        return (eval_weighted_distances, False)
    else:
        return None


class Selector:
    def __init__(self, eval_method, config=None, minimize=True, min_quantization_amount=0.0):
        self.best_value = None
        self.best_results = list()
        self.minimize = minimize
        self.metric_name = eval_method
        self.min_quantization_amount = min_quantization_amount
        self.eval_func, self.need_infer = eval_func_from_id(eval_method)

    def check_best(self, context, model, verbose=False):
        reconstructed_image = None
        if self.need_infer:
            reconstructed_image = patched_infer(model, context.grid, 2)

        value = self.eval_func(context, reconstructed_image)
        if verbose:
            print(f"Evaluation metric ({self.metric_name}): {value}")

        if value is not None and context.quantization_amount >= self.min_quantization_amount and (
            self.best_value is None
            or (self.minimize and value < self.best_value)
            or (not self.minimize and value > self.best_value)
        ):
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_optimizer_state_dict = copy.deepcopy(context.optimizer.state_dict())
            self.best_value = value
