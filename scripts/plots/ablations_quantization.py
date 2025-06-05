import scripts.plots.common as plot

bpp_range = [0.0, 2.5]

legend_fontsize = 5.5
legend_ncols = 1

experiment_id = "Kodak (ablation)"

height = 2.0
width = 5.5

legend_overrides = dict()

metric_map = {"psnr": "PSNR", "msssim": "MS-SSIM", "visual": "Visual"}

for preset in ["PSNR", "MS-SSIM", "Visual"]:
    legend_overrides[f"NIF25 [{preset}]"] = "Adaptive quantization (Ours)"

    for bits in [6, 7, 8, 12]:
        legend_overrides[f"NIF25 [{preset}] - {bits}-bits quantization"] = (
            f"{bits}-bits quantization"
        )


for metric in ["psnr", "msssim", "visual"]:
    plot.clear_results()
    plot.clear_fig()
    plot.init(1, 3)

    plot.add_results_path(f"results/summaries/kodak/inr/nif24_{metric}.json")

    plot.add_results_path(
        f"results/summaries/ablations/kodak/quantization/{metric}_06bits.json"
    )
    plot.add_results_path(
        f"results/summaries/ablations/kodak/quantization/{metric}_07bits.json"
    )
    plot.add_results_path(
        f"results/summaries/ablations/kodak/quantization/{metric}_08bits.json"
    )
    plot.add_results_path(
        f"results/summaries/ablations/kodak/quantization/{metric}_12bits.json"
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
        color_by_label=True,
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
        color_by_label=True,
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
        color_by_label=True,
        legend_overrides=legend_overrides,
        legend=False,
    )
    plot.set_dump_path(f"plots/ablations_quantization_{metric}.pdf")

    plot.save(
        height,
        width,
        legend=metric == "visual",
        legend_cols=2,
        legend_fontsize=9,
        legend_anchor=(0.5, -0.125),
        title=f"NIF25 [{metric_map[metric]} preset]",
        title_fontsize=10,
    )
