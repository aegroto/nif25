import lpips
import os
import json
import torch
import sys

from skimage import io
from losses.psnr import psnr

from pytorch_msssim import ms_ssim, ssim
from serialization import read_serialized_state_dict

from utils import load_device

def ms_ssim_reshape(tensor):
    return tensor.movedim(-1, 0).unsqueeze(0)

def lpips_reshape(tensor):
    tensor = tensor.div(255.0).sub(0.5).mul(2.0)
    return tensor.movedim(-1, 0).unsqueeze(0)

def main():
    print("Loading parameters...")
    original_file_path = sys.argv[1]
    reconstructed_file_path = sys.argv[2]
    stats_path = sys.argv[3]
    compressed_file_path = sys.argv[4]

    export_stats(original_file_path, reconstructed_file_path, stats_path, compressed_file_path)

def export_stats(original_file_path, reconstructed_file_path, stats_path, compressed_file_path):
    print("Loading device...")
    device = load_device(True)

    print("Calculating compressed state size...")
    compressed_file_size = os.stat(compressed_file_path).st_size

    print("Loading images...")
    original_image_tensor = torch.from_numpy(io.imread(original_file_path)).to(device).to(torch.float32)
    reconstructed_image_tensor = torch.from_numpy(io.imread(reconstructed_file_path)).to(device).to(torch.float32)

    pixels = original_image_tensor.nelement() / 3.0

    print("Calculating stats...")

    try:
        ssim_value = ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    except Exception as e:
        print(f"Cannot calculate SSIM: {e}")
        ssim_value = None

    try:
        ms_ssim_value = ms_ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    except Exception as e:
        print(f"Cannot calculate MS-SSIM: {e}")
        ms_ssim_value = None

    try:
        loss_fn_alex = lpips.LPIPS(net='alex')
        lpips_value = loss_fn_alex(lpips_reshape(original_image_tensor), lpips_reshape(reconstructed_image_tensor)).item()
    except Exception as e:
        print(f"Cannot calculate LPIPS: {e}")
        lpips_value = None

    stats = {
        "psnr": psnr(original_image_tensor, reconstructed_image_tensor).item(),
        "ms-ssim": ms_ssim_value,
        "lpips": lpips_value,
        "ssim": ssim_value,
        "bpp": (compressed_file_size * 8) / pixels,
    }

    print(json.dumps(stats, indent=4))

    json.dump(stats, open(stats_path, "w"), indent=4)

if __name__ == "__main__":
    main()

