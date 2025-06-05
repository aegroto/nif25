colors = {
    "NIF": "red",
    "Strumpler": "darkorange",
    "Strumpler meta-learned [2021]": "blue",
    "BPG": "cyan",
    "JPEG2000": "black",
    "Xie [2021]": "green",
    "Ballè (Factorized Prior) [2017]": "orange",
    "Ballè (Hyperprior) [2017]": "orchid",
    "COIN": "violet",
    "COIN++": "magenta",
    "ICE": "magenta",
    "JPEG": "red",
    "WebP": "gray",
    "JPEG XL": "black",
    "AVIF": "darkgoldenrod",
    # COOL-CHIC
    "COOL-CHIC": "gray",
    "COOL-CHIC v2": "olive",
    "COOL-CHIC v3.1": "orchid",
    # NIF 24
    "NIF25 [PSNR]": "royalblue",
    "NIF25 [MS-SSIM]": "darkblue",
    "NIF25 [Visual]": "darkgreen",
    "NIF25 - test": "lime",
    "NIF25 - pareto": "olive",
    "NIF25 - baseline": "black",
    # Ablations
    ## Quantization
    "Adaptive quantization (Ours)": "darkgreen",
    "6-bits quantization": (0.7, 0.3, 0.3),
    "7-bits quantization": (0.4, 0.7, 0.4),
    "8-bits quantization": (0.5, 0.5, 0.7),
    "12-bits quantization": (0.1, 0.8, 0.1),
    ## Losses
    "NIF25 [PSNR] w. L2 loss": (0.1, 0.3, 0.1),
    "NIF25 [PSNR] w. L2 + SSIM loss": (0.3, 0.1, 0.1),
    "NIF25 [MS-SSIM] w. L2 loss": (0.1, 0.3, 0.7),
    "NIF25 [MS-SSIM] w. L2 + SSIM loss": (0.3, 0.1, 0.7),
    ## Traditional SIREN layers
    "NIF25 [PSNR] w. Traditional SIREN layers": (0.1, 0.1, 0.7),
    "NIF25 [MS-SSIM] w. Traditional SIREN layers": (0.7, 0.1, 0.1),
    "NIF25 [Visual] w. Traditional SIREN layers": (0.7, 0.7, 0.1),
}


def codec_color(name):
    try:
        return colors[name]
    except KeyError:
        print(f"WARNING: Unassigned color for {name}")
        return "white"
