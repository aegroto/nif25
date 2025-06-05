import json
import os
import sys

file = sys.argv[1]

stats = json.load(open(file, "r"))

clean_stats = list()

for key in stats:
    entry = stats[key]
    clean_stats.append({
        "id": key,
        "stats": entry[1][0]
    })

full = {
    "name": "NIF2 - test",
    "type": "inr",
    "results": {
        "state_bpp": [],
        "bpp": [],
        "psnr": [],
        "ms-ssim": [],
        "ssim": []
    }
}

for stat in sorted(clean_stats, key=lambda s: s["stats"]["bpp"]):
    stat["stats"]["bpp"] /= 4.5
    stat["stats"]["state_bpp"] /= 4.5

    for key in full["results"]:
        full["results"][key].append(stat["stats"][key])

json.dump(full, open(sys.argv[2], "w"))

