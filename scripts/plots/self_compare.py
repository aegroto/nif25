import math
import scripts.plots.common as plot

bpp_range = None # [0.0, 1.1]

plot.init(3, 1)

# plot.add_results_path(f"../../results/summaries/kodak/tmp/nif_sample.json")
# plot.add_results_path(f"../../results/summaries/kodak/tmp/nif_baseline.json")
# plot.add_results_path(f"../../results/summaries/kodak/tmp/webp_reduced.json")
# plot.add_results_path(f"../../results/summaries/kodak/tmp/bpg_reduced.json")

# plot.add_results_path(f"results/summaries/kodak/inr/nif2_psnr.json")
# plot.add_results_path(f"results/summaries/kodak/inr/nif2_ms_ssim.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/nif2_psnr_reduced.json")


# plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v1.json")
# plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v2.json")
# plot.add_results_path(f"results/summaries/kodak/inr/cool-chic-v3.1.json")

# plot.add_results_path(f"results/summaries/kodak/inr/nif24_psnr.json")
# plot.add_results_path(f"results/summaries/kodak/inr/nif24_msssim.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/compressed.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/compressed_pareto.json")

# plot.add_results_path(f"results/summaries/kodak/tmp/fitted.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/fitted_baseline.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/fitted_pareto.json")

# plot.add_results_path(f"results/summaries/kodak/tmp/fitted.json")
# plot.add_results_path(f"results/summaries/kodak/tmp/fitted_baseline.json")

# plot.add_results_path(f"results/summaries/kodak/traditional/avif.json")
# plot.add_results_path(f"../../results/summaries/kodak/traditional/bpg.json")
# plot.add_results_path(f"../../results/summaries/kodak/traditional/jpeg.json")
# plot.add_results_path(f"../../results/summaries/kodak/traditional/webp.json")
# plot.add_results_path(f"../../results/summaries/kodak/autoencoder/xie2021.json")

# plot.add_results_path(f"results/summaries/clic2020/nif24_msssim.json")
# plot.add_results_path(f"results/summaries/clic2020/*.json")
# plot.add_results_path(f"results/summaries/clic2020/tmp/compressed.json")
plot.add_results_path(f"results/summaries/clic2020/tmp/fitted_all.json")
plot.add_results_path(f"results/summaries/clic2020/tmp/fitted_baseline.json")
plot.add_results_path(f"results/summaries/clic2020/tmp/fitted.json")
# plot.add_results_path(f"results/summaries/clic2020/tmp/baseline_compressed.json")


plot.plot_metric(0, 0, "", "psnr", "PSNR (dB)", bpp_range, None, 1, 1, 
                 legend_ncols=2)
plot.plot_metric(0, 1, "", "ms-ssim", "MS-SSIM", bpp_range, None, 1, 1, 
                 normalizer = plot.MS_SSIM_NORM, legend_ncols=2)
plot.plot_metric(0, 2, "", "lpips", "LPIPS", bpp_range, None, 1, 1, legend_ncols=2, legend_loc = "upper right")
plot.clear_results()

plot.set_dump_path(f"plots/self_compare.pdf")
plot.save(7.0, 4.0)
