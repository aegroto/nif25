import bjontegaard
import math
import json
import argparse


def normalize_msssim(values):
    return [-10.0 * math.log(1 - value, 10) for value in values]


def normalize_lpips(values):
    return [-10.0 * math.log(value, 10) for value in values]


def bdrate(target, test, metric, normalizer=None):
    target_bpp = target["results"]["bpp"]
    test_bpp = test["results"]["bpp"]

    try:
        target_values = target["results"][metric]
        test_values = test["results"][metric]
    except KeyError:
        return "-"

    if normalizer:
        target_values = normalizer(target_values)
        test_values = normalizer(test_values)

    value = bjontegaard.bd_rate(
        target_bpp,
        target_values,
        test_bpp,
        test_values,
        method="akima",
        require_matching_points=False,
        min_overlap=0,
    )

    color = "OliveGreen" if value < 0.0 else "BrickRed"
    return f"\\textcolor{{{color}}}{{{value:.2f}\\%}}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psnr_target_stats")
    parser.add_argument("--ms_ssim_target_stats")
    parser.add_argument("--lpips_target_stats")
    parser.add_argument("--test_stats", action="append")
    args = parser.parse_args()

    psnr_target_stats = json.load(open(args.psnr_target_stats, "r"))
    ms_ssim_target_stats = json.load(open(args.ms_ssim_target_stats, "r"))
    lpips_target_stats = json.load(open(args.lpips_target_stats, "r"))

    print("\\hline")
    print(f"\\multicolumn{{4}}{{|c|}}{{BD-Rate vs. [\\%]}}\\\\")
    print("\\hline")

    out = f"{"Anchor":<25}"
    out += f"& {"PSNR":<20}"
    out += f"& {"MS-SSIM":<20}"
    out += f"& {"LPIPS":<20}"
    out += "\\\\"

    print(out)
    print("\\hline")

    for path in args.test_stats:
        if path.startswith("Header:"):
            header_text = path.replace("Header:", "")
            print(f"\\multicolumn{{4}}{{|c|}}{{{header_text}}}\\\\")
            print("\\hline")
            continue

        if path.startswith("Separate:"):
            name, psnr_path, ms_ssim_path, lpips_path = path.replace(
                "Separate:", ""
            ).split(",")

            out = f"{name:<25}"
            out += f"&{bdrate(json.load(open(psnr_path, "r")), psnr_target_stats, "psnr"):<20}"
            out += f"&{bdrate(json.load(open(ms_ssim_path, "r")), ms_ssim_target_stats, "ms-ssim", normalize_msssim):<20}"
            out += f"&{bdrate(json.load(open(lpips_path, "r")), lpips_target_stats, "lpips", normalize_lpips):<20}"
            out += "\\\\"

            print(out)
            print("\\hline")
            continue

        test_stats = json.load(open(path, "r"))
        out = f"{test_stats["name"]:<25}"
        out += f"&{bdrate(test_stats, psnr_target_stats, "psnr"):<20}"
        out += f"&{bdrate(test_stats, ms_ssim_target_stats, "ms-ssim", normalize_msssim):<20}"
        out += (
            f"&{bdrate(test_stats, lpips_target_stats, "lpips", normalize_lpips):<20}"
        )
        out += "\\\\"

        print(out)
        print("\\hline")


if __name__ == "__main__":
    main()
