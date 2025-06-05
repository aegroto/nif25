import scripts.plots.common as plot

bpp_range = [0.0, 1.7]

legend_fontsize = 5
legend_ncols = 1
legend_overrides = {
    "NIF25 [PSNR]": "NIF25 [PSNR] w. LogCosh loss (Ours)",
    "NIF25 [MS-SSIM]": "NIF25 [MS-SSIM] w. L1+SSIM loss (Ours)",
    "NIF25 [Visual]": "NIF25 [Visual] w. LogCosh+SSIM loss (Ours)",
}

experiment_id = "Kodak (ablation)"

height = 2.0
width = 5.5

metric_map = {
    "psnr": "PSNR",
    "msssim": "MS-SSIM",
}

for metric in ["psnr", "msssim"]:
    plot.clear_results()
    plot.clear_fig(clear_legend=False)
    plot.init(1, 3)

    plot.add_results_path(f"results/summaries/kodak/inr/nif24_psnr.json")
    plot.add_results_path(f"results/summaries/kodak/inr/nif24_msssim.json")
    plot.add_results_path(f"results/summaries/kodak/inr/nif24_visual.json")

    plot.add_results_path(f"results/summaries/ablations/kodak/losses/{metric}_l2.json")
    plot.add_results_path(
        f"results/summaries/ablations/kodak/losses/{metric}_l2_ssim.json"
    )

    plot.plot_metric(
        0,
        0,
        "PSNR (dB, ↑)",
        "psnr",
        "",
        bpp_range,
        None,
        1,
        1,
        legend_overrides=legend_overrides,
        legend=False,
    )
    plot.plot_metric(
        0,
        1,
        "MS-SSIM (Normalized, ↑)",
        "ms-ssim",
        "",
        bpp_range,
        [8.0, 23.0],
        1,
        1,
        normalizer=plot.MS_SSIM_NORM,
        legend_overrides=legend_overrides,
        legend=False,
    )
    plot.plot_metric(
        0,
        2,
        "LPIPS (↓)",
        "lpips",
        "",
        bpp_range,
        None,
        1,
        1,
        legend_overrides=legend_overrides,
        legend=False,
    )
    plot.set_dump_path(f"plots/ablations_loss_{metric}.pdf")
    plot.save(
        height,
        width,
        legend=metric == "msssim",
        legend_cols=2,
        legend_fontsize=8,
        legend_anchor=(0.5, -0.17),
        title=f"NIF25 [{metric_map[metric]} preset]",
        title_fontsize=10,
    )
