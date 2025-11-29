import itertools
import os
import yaml

from utils import replace_config


def main():
    if True:
        dataset = "kodak"
        for (loss_template, ) in itertools.product(
            ["l1_ssim", "mse", "visual"]
        ):
            for file in os.listdir(f"configurations/.nif/{dataset}/"):
                base_conf = yaml.safe_load(open(f"configurations/.nif/{dataset}/{file}", "r"))

                loss_template_conf = yaml.safe_load(open(f"configurations/.sub_templates/losses/{loss_template}.yaml", "r"))

                conf = dict()
                replace_config(conf, loss_template_conf)
                replace_config(conf, base_conf)

                conf_name = file.replace(".yaml", "")
                yaml.safe_dump(
                    conf, open(f"configurations/.tuning/{conf_name}_{loss_template}.yaml", "w")
                )

if __name__ == "__main__":
    main()
