import pandas
import sys
import os
import csv

folder = sys.argv[1]

def load_text_value(csv_reader, tag):
    for row in csv_reader:
        if row[2] == tag:
            return row[3]

stats = list()

for experiment in os.listdir(folder):
    try:
        scalars_csv = csv.reader(open(f"{folder}/{experiment}/scalars.csv"))
        scalars = dict()
        next(scalars_csv)
        for row in scalars_csv:
            tag = row[2]
            value = row[3]
            if tag not in scalars:
                scalars[tag] = list()
            scalars[tag].append(float(value))

        text = csv.reader(open(f"{folder}/{experiment}/text.csv"))

        # experiment_params = [configuration_file["model"][param] for param in params]

        # Stats calculation 
        bpp = float(load_text_value(text, "bpp").replace(" bits per pixel", ""))
        psnr = max(scalars["PSNR_best"])
        stats.append([
            experiment, 
            bpp, 
            psnr 
        ])
    except Exception as e:
        print(f"{experiment} error: {e}")

df = pandas.DataFrame(stats, columns=["experiment","bpp","psnr"])
print(df.sort_values("psnr").to_string())
