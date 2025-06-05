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

for stat in sorted(clean_stats, key=lambda s: s["stats"]["psnr"]):
    print(stat)

