import itertools
import numpy
import os
import yaml
import random
import copy
import hashlib

from utils import load_configuration, replace_config


def main():
    if False:
        for (loss_template, quant_bits) in itertools.product(
            ["visual"],
            [6, 7, 8, 12]
        ):
            for file in os.listdir("configurations/.nif/kodak/"):
                base_conf = yaml.safe_load(open(f"configurations/.nif/kodak/{file}", "r"))

                loss_template_conf = yaml.safe_load(open(f"configurations/.sub_templates/losses/{loss_template}.yaml", "r"))
                fixed_quant_conf = yaml.safe_load(open(f"configurations/.sub_templates/8_bit_quant.yaml", "r"))

                conf = dict()
                replace_config(conf, loss_template_conf)
                replace_config(conf, fixed_quant_conf)
                replace_config(conf, base_conf)

                entropy = float(quant_bits) - 1.0
                conf["quantization_search"]["max_symbol_search"]["ref_entropy"] = entropy
                conf["quantization_search"]["max_symbol_search"]["first_interval"] = [entropy, entropy]

                conf_name = file.replace(".yaml", "")
                yaml.safe_dump(
                    conf, open(f"configurations/.tuning/{conf_name}_{loss_template}_{quant_bits:02d}.yaml", "w")
                )

    if False:
        for (range_start, range_end, e, length) in itertools.product(
                [368, 400],
                [80, 96],
                [5.0],
                [5]
            ):
                base_conf = yaml.safe_load(open(f"configurations/default.yaml", "r"))

                base_conf["model"]["genesis_params"]["architecture"]["range"] = [range_start, range_end]
                base_conf["model"]["genesis_params"]["architecture"]["e"] = e
                base_conf["model"]["genesis_params"]["architecture"]["length"] = length

                conf_name = f"{range_start}_{range_end}_{e}_{length}"

                yaml.safe_dump(base_conf, open(f"configurations/.staging/{conf_name}_msssim.yaml", "w"))
        
    if True:
        for file in os.listdir("configurations/.staging/"):
            for (eval_mode, tolerance) in itertools.product(
                    ["ms-ssim"],
                    [0.30, 0.40, 0.50, 0.60, 0.70]
                ):
                base_conf = yaml.safe_load(open(f"configurations/.staging/{file}", "r"))

                # if eval_mode == "ms_ssim":
                #     tolerance = tolerance * 0.1

                # base_conf["quantization_search"]["eval_mode"] = eval_mode 
                base_conf["quantization_search"]["drop_tolerance"] = tolerance

                conf_base_name = file.replace('.yaml', '') 
                conf_name = f"{conf_base_name}_{tolerance:.2f}"
                yaml.safe_dump(base_conf, open(f"configurations/.tuning/{conf_name}.yaml", "w"))

if __name__ == "__main__":
    main()
