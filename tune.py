import argparse
import debug
import copy
import torch

import torch
import sys

from skimage import io

from context import initialize_training_context
from filewise_export_stats import export_stats
from infer import infer
from input_encoding import generate_grid
from models.nif import NIF
from models.nif.utils import load_image_data_into_params
from phases.fitting import fit_with_config
from phases.infer import get_padding_for_shuffling
from phases.qat import initialize_quantizers, recalibrate_quantizers 

from utils import add_margin, dump_model_stats, load_configuration, load_device, set_reproducibility
from PIL import Image


def main():
    set_reproducibility()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("base_state_dict_path", type=str)
    parser.add_argument("optimizer_state_path", type=str)
    parser.add_argument("tuned_model_dump_path", type=str)
    parser.add_argument("--infer_path", type=str, default=None)
    parser.add_argument("--stats_path", type=str, default=None)
    args = parser.parse_args()

    base_state_dict = torch.load(args.base_state_dict_path, map_location="cpu")
    optimizer_state_dict = torch.load(args.optimizer_state_path, map_location="cpu")

    tune(args.config_path, args.file_path, base_state_dict, optimizer_state_dict, args.tuned_model_dump_path,
         args.infer_path, args.stats_path)

def tune(config_path, file_path, base_state_dict, optimizer_state_dict=None, model_dump_path=None, infer_path=None, stats_path=None):
    print("Loading configuration...")
    device = load_device()

    config = load_configuration(config_path)

    metadata = copy.deepcopy(base_state_dict["__meta"])
    del base_state_dict["__meta"]

    if config["debug"]:
        debug.init_writer(f"runs/{config_path}_{file_path}_tuning")

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
    model.load_state_dict(base_state_dict)

    print(model)
    dump_model_stats(model, width, height, debug.WRITER)

    context = initialize_training_context(config["tuning"], model)
    if optimizer_state_dict:
        context.optimizer.load_state_dict(optimizer_state_dict)
    initialize_quantizers(config["tuning"], model)
    # recalibrate_quantizers(model)
    best_value = fit_with_config(context, config["tuning"], model, verbose=True, writer=debug.WRITER)

    print(f"Best value: {best_value}")

    print("Applying quantization...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if not hasattr(module, "apply_quantization"):
                continue
            module.apply_quantization()

    final_state_dict = copy.deepcopy(model.state_dict())
    final_state_dict["__meta"] = metadata

    if model_dump_path:
        print("Model weights dump...")
        model.eval()
        torch.save(final_state_dict, model_dump_path)

    if infer_path is not None:
        rescaled_reconstructed_image = infer(config_path, final_state_dict)
        io.imsave(infer_path, rescaled_reconstructed_image)

    if stats_path is not None:
        export_stats(file_path, infer_path, stats_path, model_dump_path)

    return final_state_dict

if __name__ == "__main__":
    main()
