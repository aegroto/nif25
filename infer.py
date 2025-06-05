import torchvision
import torch
from skimage import io
import sys
from input_encoding import generate_grid

from models.nif import NIF
from phases.infer import get_padding_for_shuffling, patched_forward, patched_infer
from utils import load_configuration, load_device

def load_flattened_state(model, state_dict):
    for key in list(state_dict.keys()):
        print(key)
        if key not in model.state_dict():
            del state_dict[key]
            continue

        state_dict[key] = state_dict[key].reshape(model.state_dict()[key].shape)

    model.load_state_dict(state_dict, strict=False)

def main():
    torch.random.manual_seed(1337)

    config_path = sys.argv[1]
    state_dict_path = sys.argv[2]
    reconstructed_image_path = sys.argv[3]

    state_dict = torch.load(state_dict_path)
    rescaled_reconstructed_image = infer(config_path, state_dict)
    io.imsave(reconstructed_image_path, rescaled_reconstructed_image)

def infer(config_path, state_dict):
    device = load_device()

    print("Loading configuration...")
    config = load_configuration(config_path)

    metadata = state_dict["__meta"]
    width = metadata["width"]
    height = metadata["height"]

    padding = get_padding_for_shuffling(height, width, 16)

    print("Loading model...")

    params = config["model"]
    params["generator_params"]["height"] = height + padding[3]
    params["generator_params"]["width"] = width + padding[1]
    model = NIF(**params, device=device).to(device)
    load_flattened_state(model, state_dict)
    model.eval()

    with torch.no_grad():
        # TODO: Adapt to support cropping
        model.generator.set_shuffle(model.generator.get_shuffling())
        model.generator.set_accumulation_shuffle(1)
        input_batches = model.generator.generate_input()
        uncropped_reconstructed_image = patched_forward(model, input_batches, model.generator.get_shuffling())
        width_padding = (uncropped_reconstructed_image.size(-1) - width) // 2
        height_padding = (uncropped_reconstructed_image.size(-2) - height) // 2
        reconstructed_image = torchvision.transforms.functional.crop(
            uncropped_reconstructed_image,
            0, 0,
            height, width 
        )

    rescaled_reconstructed_image = model.generator.target_postprocessor(reconstructed_image) \
        .clamp(0.0, 1.0) \
        .detach().mul(255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    return rescaled_reconstructed_image

if __name__ == "__main__":
    main()
