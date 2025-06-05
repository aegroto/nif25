import matplotlib.pyplot as pyplot
import scripts.plots.common as plot

bpp_range = [0.0, 1.85]
max_bpp = 1.85

legend_fontsize = 8.0
legend_anchor = (0.5, -0.08)
legend_overrides = {
    "NIF25 [PSNR]": "NIF25 [PSNR] $\\bf{(Ours)}$",
    "NIF25 [MS-SSIM]": "NIF25 [MS-SSIM] $\\bf{(Ours)}$",
    "NIF25 [Visual]": "NIF25 [Visual] $\\bf{(Ours)}$",
}


def plot_psnr(range):
    plot.plot_metric(
        0,
        0,
        "PSNR (dB, ↑) - Kodak",
        "psnr",
        "",
        bpp_range,
        range,
        1,
        1,
        legend=False,
        legend_overrides=legend_overrides,
        max_bpp=max_bpp,
    )


def plot_ms_ssim(range):
    plot.plot_metric(
        0,
        1,
        "MS-SSIM (Normalized, ↑) - Kodak",
        "ms-ssim",
        "",
        bpp_range,
        range,
        1,
        1,
        normalizer=plot.MS_SSIM_NORM,
        legend=False,
        legend_overrides=legend_overrides,
        max_bpp=max_bpp,
    )


def plot_lpips(range):
    plot.plot_metric(
        0,
        2,
        "LPIPS (↓) - Kodak",
        "lpips",
        "",
        None,
        range,
        1,
        1,
        legend=False,
        legend_overrides=legend_overrides,
        max_bpp=max_bpp,
    )


height = 2.5
width = 10.0

plot.init(1, 3)

plot.add_results_path(f"results/summaries/kodak/inr/nif24_psnr.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_msssim.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_visual.json")

plot.add_results_path(f"results/summaries/kodak/inr/nif.json")
plot.add_results_path(f"results/summaries/kodak/inr/ice.json")
plot.add_results_path(f"results/summaries/kodak/inr/strumpler_basic_8bit.json")
plot.add_results_path(f"results/summaries/kodak/inr/coin.json")

plot_psnr([22.0, 37.0])
plot_ms_ssim([6.0, 21.5])
plot_lpips(None)

plot.set_dump_path(f"plots/kodak_inr.pdf")
plot.save(
    height,
    width,
    legend=True,
    legend_cols=4,
    legend_fontsize=legend_fontsize,
    legend_anchor=legend_anchor,
)

plot.clear_fig(clear_legend=True)
plot.clear_results()
plot.init(1, 3)

plot.add_results_path(f"results/summaries/kodak/inr/nif24_psnr.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_msssim.json")
plot.add_results_path(f"results/summaries/kodak/inr/nif24_visual.json")

plot.add_results_path(f"results/summaries/kodak/traditional/jpeg.json")
plot.add_results_path(f"results/summaries/kodak/traditional/avif.json")
plot.add_results_path(f"results/summaries/kodak/traditional/jxl.json")

plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v1.json")
plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v2.json")
plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v3.1.json")

plot_psnr([25.0, 43.0])
plot_ms_ssim([8.0, 23.0])
plot_lpips(None)

plot.clear_results()
plot.set_dump_path(f"plots/kodak_other.pdf")
plot.save(
    height,
    width,
    legend=True,
    legend_cols=7,
    legend_fontsize=legend_fontsize,
    legend_anchor=legend_anchor,
)
