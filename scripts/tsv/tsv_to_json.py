import json
import statistics
import csv
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

configs = dict()

with open(input_file) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    skip = True
    for row in rd:
        print(row)
        if skip:
            skip = False
            continue

        lmbda = row[1]
        bpp = float(row[2])
        psnr = float(row[3]) # Change based on file

        if lmbda not in configs:
            configs[lmbda] = {
                "bpp": [],
                "psnr": [],
            }

        configs[lmbda]["bpp"].append(bpp)
        configs[lmbda]["psnr"].append(psnr)

summary = {
    "name": "Unknown",
    "type": "inr",
    "results": {
        "config": [],
        "bpp": [],
        "psnr": []
    }
}

results = list()
for lmbda in configs:
    result = {
        "config": None,
        "bpp": None,
        "psnr": None,
    }
    result["config"] = lmbda
    result["bpp"] = statistics.mean(configs[lmbda]["bpp"])
    result["psnr"] = statistics.mean(configs[lmbda]["psnr"])
    results.append(result)

results = sorted(results, key = lambda r: r["bpp"])
for result in results:
    for key in result:
        summary["results"][key].append(result[key])

json.dump(summary, open(output_file, "w"), indent=4) 
