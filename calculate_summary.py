import glob
import statistics
import os
import json
import argparse
import traceback


def main():
    print("Loading parameters...")

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("results_folder", type=str)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--codec_name", type=str, default="NIF25 - test")
    args = parser.parse_args()

    summary = {
        "name": args.codec_name,
        "type": "inr",
        "results": {
            "config": list(),
            "state_bpp": list(),
            "bpp": list(),
            "psnr": list(),
            "ms-ssim": list(),
            "lpips": list(),
            "ssim": list(),
            "encode_time": list(),
            "decode_time": list(),
            "decode_time_no_context": list(),
        },
        "results_std": dict(),
    }

    # test = list()

    results = list()

    for root_folder_name in os.listdir(args.results_folder):
        try:
            if args.filter and args.filter not in root_folder_name:
                continue

            root_folder = f"{args.results_folder}/{root_folder_name}"

            stats = {
                "bpp": list(),
                "psnr": list(),
                "ms-ssim": list(),
                "lpips": list(),
                "ssim": list(),
                "encode_time": list(),
                "decode_time": list(),
                "decode_time_no_context": list(),
            }

            print(f"Loading stats files in {root_folder}...")
            for subdir in os.listdir(root_folder):
                regex = f"{root_folder}/{subdir}/{args.name}".replace("[", "*").replace(
                    "]", "*"
                )
                glob_results = glob.glob(regex)
                if len(glob_results) == 0:
                    continue

                path = glob_results[0]
                file_stats = json.load(open(path, "r"))
                # file_stats["filename"] = path
                # test.append(file_stats)
                for key in stats:
                    if key not in file_stats:
                        continue
                    stats[key].append(file_stats[key])

                try:
                    time_path = f"{root_folder}/{subdir}/times.json"
                    time_stats = json.load(open(time_path, "r"))
                    for key in time_stats:
                        stats[key].append(time_stats[key])
                except:
                    pass

            print("Calculating and dumping summary...")
            mean_stats = dict()
            for key in stats:
                try:
                    mean_stats[key] = statistics.mean(stats[key])
                    mean_stats[f"{key}_std"] = statistics.stdev(stats[key])
                except Exception as ex_stats:
                    print(f"Couldn't calculate mean for stat {key}: {ex_stats}")

            mean_stats["config"] = root_folder_name

            if "psnr" not in mean_stats:
                print(f"WARNING: Broken config {root_folder_name}")
                continue

            results.append(mean_stats)
        except Exception as e:
            print(f"WARNING: Could not load {root_folder_name}: {e}")
            traceback.print_exc()

    results = sorted(results, key=lambda r: r["bpp"])

    # for s in sorted(test, key = lambda r: r["psnr"]):
    #     print(s)

    for result in results:
        for key in result:
            if "_std" in key:
                clean_key = key.replace("_std", "")
                if clean_key not in summary["results_std"]:
                    summary["results_std"][clean_key] = list()
                summary["results_std"][clean_key].append(result[key])
            else:
                summary["results"][key].append(result[key])

    json.dump(summary, open(args.output_file, "w"), indent=4)


if __name__ == "__main__":
    main()
