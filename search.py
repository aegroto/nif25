import os
import traceback
import copy
import torch
import json
import yaml
from compress import compress_state_dict
from decompress import decompress_state_dict

from filewise_export_stats import export_stats
from fit import fit
from infer import infer

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from skimage import io
from models.nif import NIF
from quantize import quantize
from serialization import deserialize_state_dict, read_serialized_state_dict, write_serialized_state_dict
from tune import tune

from utils import calculate_model_size, load_configuration, load_device, set_reproducibility

def parameters_list():
    return [
        range_parameter("peak_lr", [0.0, 1.0]),
        range_parameter("floor_lr", [0.0, 1.0]),

        range_parameter("peak_decay_factor", [0.9, 1.0]),
        range_parameter("floor_decay_factor", [0.9, 1.0]),

        range_parameter("amount_start", [0.0, 1.0]),
        range_parameter("amount_end", [0.0, 1.0]),
        range_parameter("amount_smoothing", [0.9, 1.0]),

        range_parameter("range_start", [0.0, 1.0]),
        range_parameter("range_end", [0.0, 1.0]),
        range_parameter("range_smoothing", [0.9, 1.0]),
    ]

def norm_param(value, min, max):
    return min + value * (max - min)

BASE_CONF = load_configuration("configurations/default.yaml")
def config_from_parameters(parameters):
    config = copy.deepcopy(BASE_CONF)

    config["fitting"]["training"]["optimizer"]["lr"] = norm_param(parameters["peak_lr"], 1.0e-4, 1.0e-2)
    config["fitting"]["training"]["scheduler"]["floor_lr"] = norm_param(parameters["floor_lr"], 1.0e-9, 1.0e-2)
    config["fitting"]["training"]["scheduler"]["peak_decay_factor"] = parameters["peak_decay_factor"]
    config["fitting"]["training"]["scheduler"]["floor_decay_factor"] = parameters["floor_decay_factor"]

    for cat in ["amount", "range"]:
        for param in ["start", "end", "smoothing"]:
            config["fitting"]["restart"][cat][param] = parameters[f"{cat}_{param}"]

    # config["quantization"]["training"]["optimizer"]["lr"] = norm_param(parameters["quant_peak_lr"], 1.0e-6, 1.0e-2)
    # config["quantization"]["training"]["scheduler"]["floor_lr"] = norm_param(parameters["quant_floor_lr"], 1.0e-8, 1.0e-2)

    return config

def range_parameter(name, bounds):
    return {
        "name": name, 
        "type": "range",
        "bounds": bounds
    }

def choice_parameter(name, values):
    return {
        "name": name, 
        "type": "choice",
        "values": values
    }

INVALID_RESULT = { "bpp": 32.0, "psnr": 0.0 }

def estimate_uncompressed_bpp(conf): 
    device = torch.device("cuda")
    model = NIF(input_features=2, **conf["model"], device=device)
    model_size = calculate_model_size(model, verbose=False)
    pixels_count = 768 * 512
    bpp = (model_size * 8) / pixels_count
    return bpp

def evaluate(parameters):
    config = config_from_parameters(parameters)
    # bpp = estimate_uncompressed_bpp(config)
    # if bpp > 8.0:
    #     print(f"Invalid bpp: {bpp}")
    #     return { "state_bpp": bpp, "psnr": 1.0 }

    config_path = "configurations/__search.yaml"
    yaml.safe_dump(config, open(config_path, "w"))

    def run_for(image_id):
        original_file_path = f"test_images/kodak/{image_id}.png"
        results_folder = f"results/nif/__search/{image_id}/"
        base_state_path = f"{results_folder}/state.pth"
        fp_quantized_state_path = f"{results_folder}/fp_state.pth"
        optimizer_state_path = f"{results_folder}/optimizer_state.pth"
        compressed_path = f"{results_folder}/compressed.nif"
        decoded_path = f"{results_folder}/decoded.png"
        stats_path = f"{results_folder}/stats.json"

        try:
            os.mkdir(results_folder)
        except FileExistsError:
            pass

        set_reproducibility()
        try:
            _, best_eval_value = fit(config_path, original_file_path, return_best_value=True)
            return {
                "best_value": best_eval_value
            }
            # _, _ = quantize(config_path, original_file_path, fitted_state_dict,
            #                 full_precision_quantized_model_dump_path=fp_quantized_state_path)
            # compressed_state_dict = compress_state_dict(config_path, deserialize_state_dict(fp_quantized_state_path))
            # write_serialized_state_dict(compressed_state_dict, compressed_path)

            # compressed_state_dict = read_serialized_state_dict(compressed_path)
            # decompressed_state_dict = decompress_state_dict(config_path, compressed_state_dict, device=load_device())

            # reconstructed_image = infer(config_path, decompressed_state_dict)
            # io.imsave(decoded_path, reconstructed_image)

            # if stats_path is not None:
            #     export_stats(original_file_path, decoded_path, stats_path, compressed_path)

            # stats = json.load(open(stats_path, "r"))
            # for key in list(stats.keys()):
            #     if stats[key] is None:
            #         del stats[key]

            # return stats
        except AttributeError as e:
            traceback.print_exc()
            # best_value = 1.0e+2
            return INVALID_RESULT

        # return {
        #     "best_value": best_value
        # }

        # base_state_dict = torch.load(base_state_path)
        # optimizer_state_dict = torch.load(optimizer_state_path)
        # tuned_state_dict = tune(config_path, original_file_path, base_state_dict, optimizer_state_dict)

        # fp_quantized_state_dict, _ = quantize(config_path, original_file_path, tuned_state_dict)

        # rescaled_reconstructed_image = infer(config_path, fp_quantized_state_dict)
        # rescaled_reconstructed_image = infer(config_path, tuned_state_dict)
        # io.imsave(reconstructed_file_path, rescaled_reconstructed_image)

        # export_stats(original_file_path, reconstructed_file_path, stats_path, base_state_path)

        # stats = json.load(open(stats_path, "r"))
        # for key in list(stats.keys()):
        #     if stats[key] is None:
        #         del stats[key]

        # return stats

    all_stats = list()
    for image_id in [24, 1, 7, 8, 20]:
    # for image_id in [24, 7, 8]:
    # for image_id in [24]:
    # for image_id in [
    #     "0f89df7638bc485db94a981367ad6983bda3396e153893cf80794790e48d3df7",
    #     "2ff7069b3e9ba2e7a1aaf400783004d0dfcc762cdab00b8be922fe6b685d85ea",
    #     "03bcfef063be6a7db416b1cf8c227f201d6a6b7c2aaee7200a46c96d2b4c4f37"
    # ]:
        all_stats.append(run_for(image_id))

    mean_stats = dict()
    for key in list(all_stats[0].keys()):
        total = 0.0
        for stats in all_stats:
            total += stats[key]
        mean_stats[key] = total / len(all_stats)

    return mean_stats

def main():
    ax_client = AxClient()
    ax_client.create_experiment(
        name="search",
        parameters=parameters_list(),
        objectives={
            # "psnr": ObjectiveProperties(minimize=False, threshold=20.0),
            # "ms-ssim": ObjectiveProperties(minimize=False, threshold=0.85),
            # "bpp": ObjectiveProperties(minimize=True, threshold=20.0),

            "best_value": ObjectiveProperties(minimize=True, threshold=1.0),
        },
    )

    while True:
        try:
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

            try:
                best_parameters = ax_client.get_pareto_optimal_parameters()
            except:
                best_parameters = ax_client.get_best_parameters()

            json.dump(best_parameters, open("results/nif/optimization.json", "w"))
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
