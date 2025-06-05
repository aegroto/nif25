import json
import os
import sys

root_folder = sys.argv[1]
experiment_id = int(sys.argv[2])

stats = list()

for folder in os.listdir(root_folder):
    path = f"{root_folder}/{folder}/{experiment_id}/stats.json"
    try:
        data = json.load(open(path, "r"))
        data["id"] = folder
        stats.append(data)
    except:
        print(f"Cannot load {folder}")

for stat in sorted(stats, key=lambda s: s["psnr"]):
    print(stat)

