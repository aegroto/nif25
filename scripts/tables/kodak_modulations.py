import json

root = "results/summaries/ablations/kodak_modulations"
baseline_data = json.load(open(f"{root}/baseline.json"))
no_modulations_data = json.load(open(f"{root}/no_modulations.json"))

for i in range(0, len(baseline_data["results"]["config"])):
    config_id = baseline_data["results"]["config"][i]

    baseline_bpp = baseline_data["results"]["state_bpp"][i]
    no_modulations_bpp = no_modulations_data["results"]["state_bpp"][i]
    bpp_decrease = no_modulations_bpp - baseline_bpp

    baseline_psnr = baseline_data["results"]["psnr"][i]
    no_modulations_psnr = no_modulations_data["results"]["psnr"][i]
    psnr_decrease = no_modulations_psnr - baseline_psnr

    print("\hline")
    print(f"{baseline_bpp:.2f}bpp / {baseline_psnr:.2f}dB & {bpp_decrease:.2f}bpp & {psnr_decrease:.2f}dB \\\\")

