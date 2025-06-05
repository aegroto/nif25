import argparse
import debug
import copy
import torch

import torch
import sys

from context import initialize_training_context
from filewise_export_stats import export_stats
from infer import infer
from input_encoding import generate_grid
from models.nif import NIF
from models.nif.utils import load_image_data_into_params
from phases.fitting import fit_with_config
from phases.infer import get_padding_for_shuffling
from phases.qat import generate_quantization_config, initialize_quantizers

from utils import add_margin, dump_model_stats, load_configuration, load_device, set_reproducibility
from PIL import Image

from skimage import io

def main():
    set_reproducibility()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("model_dump_path", type=str)
    parser.add_argument("optimizer_dump_path", type=str)
    parser.add_argument("--infer_path", type=str, default=None)
    parser.add_argument("--stats_path", type=str, default=None)
    args = parser.parse_args()

    fit(args.config_path, args.file_path, args.model_dump_path, args.optimizer_dump_path, args.infer_path, args.stats_path)

def fit(config_path, file_path, model_dump_path=None, optimizer_dump_path=None, 
        infer_path=None, stats_path=None, return_best_value=False):
    print("Loading configuration...")
    device = load_device()

    config = load_configuration(config_path)

    if config["debug"]:
        debug.init_writer(f"runs/{config_path}_{file_path}_fitting")

    debug.WRITER.add_text("config", "```\n" + str(config).replace('\n', '\n\n') + "\n```")

    print("Loading images...")
    image = Image.open(file_path)

    (height, width) = (image.size[1], image.size[0])

    padding = get_padding_for_shuffling(height, width, 16)
    image = add_margin(image, 0, padding[1], padding[3], 0, (0, 0, 0))

    (padded_height, padded_width) = (image.size[1], image.size[0])
    
    print("Loading model...")
    params = config["model"]
    params = load_image_data_into_params(params, padded_height, padded_width)
    model = NIF(**params, device=device)
    model = model.to(device)
    with torch.no_grad():
        model.generator.set_target(image)

    print(model)
    dump_model_stats(model, width, height, debug.WRITER)

    context = initialize_training_context(config["fitting"], model)
    quantization_config = generate_quantization_config(config["fitting"], model)
    initialize_quantizers(quantization_config, model)
    best_value = fit_with_config(context, config["fitting"], model, verbose=True, writer=debug.WRITER)

    # print("Applying quantization...")
    # with torch.no_grad():
    #     for _, module in model.named_modules():
    #         if not hasattr(module, "apply_quantization"):
    #             continue
    #         module.apply_quantization()

    final_state_dict = copy.deepcopy(model.state_dict())
    final_state_dict["__meta"] = {
        "width": width,
        "height": height,
    }

    if model_dump_path:
        print("Model weights dump...")
        model.eval()
        torch.save(final_state_dict, model_dump_path)

    if optimizer_dump_path:
        print("Optimizer state dump...")
        torch.save(context.optimizer.state_dict(), optimizer_dump_path)

    if infer_path is not None:
        rescaled_reconstructed_image = infer(config_path, final_state_dict)
        io.imsave(infer_path, rescaled_reconstructed_image)

    if stats_path is not None:
        export_stats(file_path, infer_path, stats_path, model_dump_path)

    if return_best_value:
        return final_state_dict, best_value
    else:
        return final_state_dict

if __name__ == "__main__":
    main()
