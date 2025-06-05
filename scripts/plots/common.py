import math
import random
import hashlib

import glob
from itertools import cycle
from matplotlib import cm
import numpy
import matplotlib.pyplot as plt

import json
import os
import sys

from matplotlib.ticker import FormatStrFormatter

from scripts.plots.colors import codec_color

MS_SSIM_NORM = lambda v: -10 * math.log10(1.0 - v)
LPIPS_NORM = lambda v: 10 * math.log10(v)

results_paths = list()
dump_path = None

def add_results_path(path):
    global results_paths
    results_paths.append(path)

def set_dump_path(path):
    global dump_path
    try:
        os.mkdir(os.path.dirname(path))
    except FileExistsError:
        pass
    dump_path = path

def codec_style(codec_type):
    if codec_type == "inr":
        return {
            "marker": "",
            "linestyle": "-",
            "linewidth": 1.25
        }
    if codec_type == "inr_other":
        return {
            "marker": "",
            "linestyle": "-",
            "linewidth": 0.75
        }
    if codec_type == "ablation_baseline":
        return {
            "marker": "",
            "linestyle": "-",
            "linewidth": 1.0,
            "markersize": 3,
        }
    if codec_type == "ablation":
        return {
            "marker": "",
            "linestyle": "--",
            "linewidth": 1.0,
            "markersize": 3,
        }
    elif codec_type == "traditional":
        return {
            "marker": "",
            "linestyle": "dotted",
            "linewidth": 1.0,
            "markersize": 1.5
        }
    elif codec_type == "autoencoder":
        return {
            "marker": "",
            "linestyle": "--",
            "linewidth": 0.75
        }
    elif codec_type == "hybrid":
        return {
            "marker": "",
            "linestyle": "--",
            "linewidth": 0.75,
            "markersize": 3.5
        }

fig = None
axes = None

def init(rows=1, cols=2, font_size=8):
    global fig
    global axes

    plt.rc('font', size=font_size)

    fig, axes = plt.subplots(rows, cols)
    fig.tight_layout()

legend_handles = list()
legend_labels = list()

def plot_metric(row, col, title, metric_name, display_name, xlim, ylim, xdigits=2, ydigits=2,
                normalizer=None, x_value="bpp", 
                legend=True, legend_fontsize=6, legend_ncols=1, legend_loc='lower right',
                legend_overrides=None,
                color_by_label=False,
                max_bpp=None):
    try:
        if hasattr(axes, "ndim") and axes.ndim == 2:
            ax = axes[row][col]
        else:
            ax = axes[col]
    except TypeError as e:
        ax = axes

    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{xdigits}f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{ydigits}f"))

    for result_path in results_paths:
        for full_path in glob.glob(result_path):
            try:
                stats = json.load(open(full_path))
                results = stats["results"]
                codec_name = stats["name"]

                bpp_values = results[x_value]
                metric_values = results[metric_name]

                if max_bpp is not None:
                    i = 0
                    while i < len(bpp_values):
                        bpp = bpp_values[i]
                        if bpp > max_bpp:
                            del bpp_values[i]
                            del metric_values[i]
                        else:
                            i += 1


                if normalizer:
                    metric_values = [normalizer(value) for value in metric_values]

                if legend_overrides is None or codec_name not in legend_overrides:
                    legend_label = codec_name
                else:
                    legend_label = legend_overrides[codec_name]

                if color_by_label:
                    color = codec_color(legend_label)
                else:
                    color = codec_color(codec_name)

                line, = ax.plot(
                    bpp_values, 
                    metric_values, 
                    color=color, 
                    **codec_style(stats["type"])
                )

                line.set_label(legend_label)

                if legend_label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(legend_label)
            except Exception as e:
                print(f"Cannot load {full_path}: {e}")

    # _, xend = ax.get_xlim()
    # _, yend = ax.get_ylim()
    # ax.xaxis.set_ticks(numpy.arange(0.0, xend, 1.0))
    # ax.yaxis.set_ticks(numpy.arange(0.0, yend, 0.1))

    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel(display_name)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

    if legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncols=legend_ncols)

    ax.grid()

# CelebA
# plot_metric(0, "psnr", "PSNR", [0.0, 3.0], [25.0, 40.0])
# plot_metric(1, "ms-ssim", "MS-SSIM", [0.0, 3.0], [0.95, 1.0])

def clear_results():
    global dump_path
    global results_paths
    results_paths.clear()
    dump_path = None

def clear_fig(clear_legend=True):
    global fig
    global axes
    if clear_legend:
        legend_handles.clear()
        legend_labels.clear()
    fig = None
    axes = None
    plt.clf()

def save(height=2.5, width=6.5, legend=False, legend_cols=2, legend_fontsize=8, legend_anchor=(0.5, 0.0), title=None, title_fontsize=8):
    global dump_path
    fig.set_figheight(height)
    fig.set_figwidth(width)

    if legend:
        fig.legend(legend_handles, legend_labels, loc='center', ncol=legend_cols, fontsize=legend_fontsize, bbox_to_anchor=legend_anchor)
    
    if title is not None:
        fig.text(0.5, 1.0, title, fontsize=title_fontsize, horizontalalignment="center")

    fig.tight_layout()
    plt.savefig(dump_path, bbox_inches="tight")
