import argparse
import copy
import json
import oapackage

parser = argparse.ArgumentParser()
parser.add_argument("stats_file_path")
parser.add_argument("pareto_stats_file_path")
args = parser.parse_args()

stats = json.load(open(args.stats_file_path, "r"))

config_names = stats["results"]["config"]
bpp_values = stats["results"]["bpp"]
psnr_values = stats["results"]["psnr"]

pairs = list(zip(bpp_values, psnr_values))

pareto=oapackage.ParetoDoubleLong()

for (idx, (bpp, psnr)) in enumerate(pairs):
    w = oapackage.doubleVector((-bpp, psnr))
    pareto.addvalue(w, idx)

pareto.show(verbose=1)
optimal_indices = pareto.allindices()

optimal_stats = {
    "name": stats["name"],
    "type": stats["type"],
    "results": {
        "config": list(),
        "bpp": list(),
        "psnr": list(),
    }
}

for i in optimal_indices:
    for key in ["config", "bpp", "psnr"]:
        optimal_stats["results"][key].append(stats["results"][key][i])

json.dump(optimal_stats, open(args.pareto_stats_file_path, "w"), indent=4)