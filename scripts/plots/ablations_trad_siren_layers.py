import scripts.plots.common as plot

bpp_range = [0.0, 1.7]

experiment_id = "Kodak (ablation)"

legend_fontsize = 5
legend_ncols = 1
legend_overrides = {
    "NIF25 [PSNR]": "NIF25 [PSNR] w. Self-modulating SIREN Layer (Ours)",
    "NIF25 [MS-SSIM]": "NIF25 [MS-SSIM] w. Self-modulating SIREN Layer (Ours)",
    "NIF25 [Visual]": "NIF25 [Visual] w. Self-modulating SIREN Layer (Ours)",
}

height = 3.0
width = 10.0

plot.clear_results()
plot.clear_fig()
plot.init(1, 3)

plot.add_results_path(f"results/summaries/kodak/inr/nif24_psnr.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_msssim.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_visual.json")

plot.add_results_path(f"results/summaries/ablations/kodak/trad_siren_layers/*.json")

plot.plot_metric(
    0,
    0,
    f"PSNR (dB, ↑) - {experiment_id}",
    "psnr",
    "",
    bpp_range,
    None,
    1,
    1,
    legend_fontsize=legend_fontsize,
    legend_ncols=legend_ncols,
    legend_overrides=legend_overrides,
)
plot.plot_metric(
    0,
    1,
    f"MS-SSIM (Normalized, ↑) - {experiment_id}",
    "ms-ssim",
    "",
    bpp_range,
    [8.0, 23.0],
    1,
    1,
    normalizer=plot.MS_SSIM_NORM,
    legend_fontsize=legend_fontsize,
    legend_ncols=legend_ncols,
    legend_overrides=legend_overrides,
)
plot.plot_metric(
    0,
    2,
    f"LPIPS (↓) - {experiment_id}",
    "lpips",
    "",
    bpp_range,
    None,
    1,
    1,
    legend_fontsize=legend_fontsize,
    legend_ncols=legend_ncols,
    legend_loc="upper right",
    legend_overrides=legend_overrides,
)
plot.set_dump_path(f"plots/ablations_trad_siren_layers.pdf")
plot.save(height, width)
