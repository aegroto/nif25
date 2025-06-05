import torchvision
from skimage import io
import copy
import torch

import yaml
import torch
import sys

from torch.utils.tensorboard import SummaryWriter
from input_encoding import generate_grid
from losses.gradient import grad
from models.nif import NIF
from phases.fitting import fit_with_config
from transform import default_inv_transform, default_transform

from utils import dump_model_stats, load_configuration, load_device
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image


def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.deterministic = True

    file_path = sys.argv[1]
    # decoded_file_path = sys.argv[2]
    output_file_path = sys.argv[2]

    print("Loading images...")
    image = Image.open(file_path)

    (height, width) = (image.size[1], image.size[0])

    transform = default_transform(width, height)
    inv_transform = default_inv_transform()

    transformed_image_tensor = transform(image)
    inv_transformed_image_tensor = inv_transform(transformed_image_tensor)

    output = grad(inv_transformed_image_tensor) 
    outputs = output.squeeze(0).unbind(1)

    for i in range(len(outputs)):
        output_image = outputs[i] \
            .clamp(0.0, 1.0) \
            .detach().mul(255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

        io.imsave(f"{output_file_path}_{i}.png", output_image)

if __name__ == "__main__":
    main()
