import functools
import logging
import glob
import json
import argparse

def load_stats(root_folder, stats_file_path):
    root_stats = dict()
    for file in glob.glob(f"{root_folder}/**/{stats_file_path}"):
        key = file.replace(root_folder, "").replace(f"/{stats_file_path}", "")
        stats = json.load(open(file, "r"))
        root_stats[key] = stats
    return root_stats

def check_integrity(pivot_stats, other_stats):
    if len(pivot_stats) != len(other_stats):
        logging.warning(f"Stats dict of different length ({len(pivot_stats)} != {len(other_stats)})")

    for key in pivot_stats:
        try:
            other_stats[key]
        except KeyError:
            logging.warning(f"Missing key {key} in other stats")

    for key in other_stats:
        try:
            pivot_stats[key]
        except KeyError:
            logging.warning(f"Missing key {key} in pivot stats")

def perc(other, pivot):
    return ((other  / pivot) - 1.0) * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pivot_folder", type=str)
    parser.add_argument("other_folder", type=str)
    parser.add_argument("pivot_stats_file_path", type=str, default="stats.json")
    parser.add_argument("other_stats_file_path", type=str, default="stats.json")
    args = parser.parse_args()

    pivot_stats_dict = load_stats(args.pivot_folder, args.pivot_stats_file_path)
    other_stats_dict = load_stats(args.other_folder, args.other_stats_file_path)

    check_integrity(pivot_stats_dict, other_stats_dict)

    percs = list()
    for key in pivot_stats_dict:
        try:
            pivot_stats = pivot_stats_dict[key]
            other_stats = other_stats_dict[key]

            stat_name = "ms-ssim"

            stat_diff = other_stats[stat_name] - pivot_stats[stat_name] 
            bpp_diff = other_stats["bpp"] - pivot_stats["bpp"]

            stat_perc = perc(other_stats[stat_name], pivot_stats[stat_name])
            bpp_perc = perc(other_stats["bpp"], pivot_stats["bpp"])

            percs.append((stat_perc, bpp_perc))
            
            print(f"{key[:16]}: \t{stat_name}: {stat_diff:+.3f} ({stat_perc:+.2f}%), bpp: {bpp_diff:+.5f} ({bpp_perc:+.2f}%)")
            # diffs.append((key[:4], stat_diff, bpp_diff))
        except Exception as e:
            print(f"Cannot compare {key}: {repr(e)}")

    def mean(idx):
        sum = functools.reduce(lambda result, pair: result + pair[idx], percs, 0)
        return sum / len(percs)

    print(f"Mean gains -- {stat_name.upper()}: {mean(0):+.2f}%, bpp: {mean(1):+.2f}%")

    # for entry in sorted(diffs, key=lambda x: x[1]):
    #     print(f"{entry[0]}: \tPSNR: {entry[1]:+.3f} (), bpp: {entry[2]:+.5f}")

if __name__ == "__main__":
    main()