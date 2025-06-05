import torch
from pytorch_msssim import ms_ssim

from losses.psnr import psnr
from phases.infer import patched_forward

def calculate_eval_value(model, mode="psnr"):
    model.eval()

    current_shuffle = model.generator.shuffle
    current_accumulation_shuffle = model.generator.accumulation_shuffle

    model.generator.set_shuffle(model.generator.get_shuffling())
    model.generator.set_accumulation_shuffle(1)
    input_batches = model.generator.generate_input()

    with torch.no_grad():
        reconstructed_image = patched_forward(model, input_batches, model.generator.get_shuffling())


    model.generator.set_shuffle(current_shuffle)
    model.generator.set_accumulation_shuffle(current_accumulation_shuffle)

    postprocessed_image = model.generator.target_postprocessor(reconstructed_image).unsqueeze(0)
    postprocessed_target = model.generator.target_postprocessor(model.generator.target_tensor).unsqueeze(0)

    if mode == "psnr":
        eval_value = psnr(postprocessed_target, postprocessed_image, 1.0).item()
    elif mode == "ms-ssim":
        eval_value = ms_ssim(postprocessed_target, postprocessed_image, 1.0).item()
    else:
        eval_value = None

    # eval_value = 20 * math.log10(1.0 / math.sqrt(best_loss))

    model.train()
    return eval_value
