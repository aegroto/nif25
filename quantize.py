import numpy
import argparse
import debug
import copy
import torch

import torch

from PIL import Image
from skimage import io

from context import initialize_training_context
from filewise_export_stats import export_stats
from infer import infer
from models.nif import NIF
from models.nif.utils import load_image_data_into_params
from phases.fitting import fit_with_config
from phases.infer import get_padding_for_shuffling
from phases.qat import apply_quantization, generate_quantization_config, initialize_quantizers, recalibrate_quantizers, set_quantization_max_symbol 

from torch.utils.tensorboard import SummaryWriter
from phases.qat.max_symbol_search import find_best_max_symbols, qat_with_config
from phases.qat.utils import apply_max_symbols_in_config, extract_max_symbols
from serialization import serialize_state_dict

from utils import add_margin, dump_model_stats, load_configuration, load_device, replace_config, set_reproducibility, setup_logging


def main():
    setup_logging()
    set_reproducibility()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("unquantized_model_path", type=str)
    parser.add_argument("quantized_model_dump_path", type=str)
    parser.add_argument("fp_quantized_model_dump_path", type=str)
    parser.add_argument("--optimizer_state_path", type=str, required=False)
    parser.add_argument("--infer_path", type=str, default=None)
    parser.add_argument("--stats_path", type=str, default=None)
    args = parser.parse_args()

    unquantized_state_dict = torch.load(args.unquantized_model_path, map_location="cpu")

    optimizer_state_dict = None
    if args.optimizer_state_path is not None:
        optimizer_state_dict = torch.load(args.optimizer_state_path)

    quantize(
        args.config_path,
        args.file_path,
        unquantized_state_dict,
        optimizer_state_dict,
        args.quantized_model_dump_path,
        args.fp_quantized_model_dump_path,
        args.infer_path,
        args.stats_path,
    )


def quantize(
    config_path,
    file_path,
    unquantized_state_dict,
    optimizer_state_dict=None,
    quantized_model_dump_path=None,
    full_precision_quantized_model_dump_path=None,
    infer_path=None,
    stats_path=None,
):
    device = load_device()

    print("Loading configuration...")
    config = load_configuration(config_path)

    metadata = copy.deepcopy(unquantized_state_dict["__meta"])
    del unquantized_state_dict["__meta"]

    if config["debug"]:
        debug.init_writer(f"runs/{config_path}_{file_path}_quantization")

    debug.WRITER.add_text("config", "```\n" + str(config).replace("\n", "\n\n") + "\n```")

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

    dump_model_stats(model, width, height, debug.WRITER)

    best_max_symbols = find_best_max_symbols(config, model, unquantized_state_dict)

    print(f"Fine-tuning with best config value {list(best_max_symbols.values())}")

    torch.cuda.empty_cache()

    best_config = generate_quantization_config(config["quantization"], model)
    apply_max_symbols_in_config(best_max_symbols, best_config)
    best_eval_value = qat_with_config(config["quantization"], model, unquantized_state_dict, best_config)

    print("Applying quantization...")
    apply_quantization(model)

    print("Loading best state...")
    best_full_precision_state_dict = copy.deepcopy(model.state_dict())
    best_full_precision_state_dict["__meta"] = metadata

    if infer_path is not None:
        infer_state_dict = copy.deepcopy(best_full_precision_state_dict)
        rescaled_reconstructed_image = infer(config_path, infer_state_dict)
        io.imsave(infer_path, rescaled_reconstructed_image)

    print("Model weights dump...")

    quantization_config = dict()

    for name, module in model.named_modules():
        try:
            if hasattr(module, "quantizer"):
                module_config = module.quantizer.get_config()
                parameter_name = f"{name}.{module.quantizer.id}"

                quantization_config[parameter_name] = module_config
        except Exception as e:
            print(f"Unable to persist quantization on module {name}: {e}")

    best_full_precision_state_dict["quantization_config"] = quantization_config

    if full_precision_quantized_model_dump_path:
        serialize_state_dict(
            best_full_precision_state_dict, full_precision_quantized_model_dump_path
        )

    if quantized_model_dump_path:
        with torch.no_grad():
            print("Exporting quantized weights...")
            quantized_dict = dict()

            for name, module in model.named_modules():
                try:
                    if hasattr(module, "quantizer"):
                        quantization_result = module.quantizer.quantize_tensor(module.get_raw_values())
                        quantized_dict[name] = quantization_result
                except Exception as e:
                    print(f"Unable to apply quantization on module {name}: {e}")

            serialize_state_dict(quantized_dict, quantized_model_dump_path)

    if stats_path is not None:
        export_stats(file_path, infer_path, stats_path, full_precision_quantized_model_dump_path)

    return best_full_precision_state_dict, best_eval_value


if __name__ == "__main__":
    main()
