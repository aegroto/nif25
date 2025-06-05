import statistics
import sys
import os
import json

root_folder = sys.argv[1]

stats = dict()

for folder in os.listdir(root_folder):
    path = f"{root_folder}/{folder}/stats.json"
    try:
        data = json.load(open(path, "r"))
        for key in ["psnr", "ms-ssim"]:
            if key not in stats:
                stats[key] = list()

            stats[key].append(data[key])
    except:
        print(f"Cannot load {folder}")

def val_string(value, mean):
    return f"({value:.2f}, {(value / mean) * 100.0:.2f}%)"

for key in stats:
    values = stats[key]
    mean = statistics.mean(values)
    jitters = [abs(value - mean) for value in values]
    jitter = statistics.mean(jitters)
    range = max(values) - min(values)
    print(f"{key} - mean: {mean:.2f}, jitter: {val_string(jitter, mean)}, range: {val_string(range, mean)}")
