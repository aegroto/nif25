import scripts.plots.common as plot

plot.init(1, 3)

height = 2.5
width = 10.0
bpp_range = None  # [0.0, 1.2]
max_bpp = 0.8

legend_overrides = {
    "NIF25 [PSNR]": "NIF25 [PSNR] $\\bf{(Ours)}$",
    "NIF25 [MS-SSIM]": "NIF25 [MS-SSIM] $\\bf{(Ours)}$",
    "NIF25 [Visual]": "NIF25 [Visual] $\\bf{(Ours)}$",
}

# plot.add_results_path(f"results/summaries/clic2020/*.json")

plot.add_results_path(f"results/summaries/clic2020/nif24_msssim.json")
plot.add_results_path(f"results/summaries/clic2020/nif24_psnr.json")
plot.add_results_path(f"results/summaries/clic2020/nif24_visual.json")
plot.add_results_path(f"results/summaries/clic2020/cool-chic-v2.json")
plot.add_results_path(f"results/summaries/clic2020/cool-chic-v3.1.json")
plot.add_results_path(f"results/summaries/clic2020/avif.json")
plot.add_results_path(f"results/summaries/clic2020/jpeg.json")
plot.add_results_path(f"results/summaries/clic2020/jxl.json")
plot.add_results_path(f"results/summaries/clic2020/webp.json")
plot.add_results_path(f"results/summaries/clic2020/nif.json")

plot.plot_metric(
    0,
    0,
    "",
    "psnr",
    "PSNR (dB, ↑)",
    bpp_range,
    None,
    1,
    1,
    legend=False,
    legend_overrides=legend_overrides,
    max_bpp=max_bpp,
)
plot.plot_metric(
    0,
    1,
    "",
    "ms-ssim",
    "MS-SSIM (Normalized, ↑)",
    bpp_range,
    None,
    1,
    1,
    normalizer=plot.MS_SSIM_NORM,
    legend=False,
    legend_overrides=legend_overrides,
    max_bpp=max_bpp,
)
plot.plot_metric(
    0,
    2,
    "",
    "lpips",
    "LPIPS (↓)",
    bpp_range,
    None,
    1,
    1,
    legend=False,
    legend_overrides=legend_overrides,
    max_bpp=max_bpp,
)

plot.set_dump_path(f"plots/clic2020.pdf")
plot.save(
    height,
    width,
    legend=True,
    legend_cols=5,
    legend_fontsize=8.0,
    legend_anchor=(0.5, -0.08),
)
