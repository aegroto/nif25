import torch

from utils import pad_measure

def get_padding_for_shuffling(height, width, patching):
    height_padding = pad_measure(height, patching)
    width_padding = pad_measure(width, patching)

    padding = (0, width_padding, 0, height_padding)

    return padding

def pad_for_shuffling(tensor, patching):
    height = tensor.size(-2)
    width = tensor.size(-1)

    height_padding = pad_measure(height, patching)
    width_padding = pad_measure(width, patching)

    padding = (
        width_padding // 2, 
        width_padding // 2 + width_padding % 2, 
        height_padding // 2, 
        height_padding // 2 + height_padding % 2
    )

    padded_tensor = torch.nn.functional.pad(tensor, padding, "replicate")

    return padded_tensor

def patched_forward(model, input_batches, patching):
    reconstructed_patches = list()
    for input_batch in input_batches:
        for input_sample in input_batch:
            reconstructed_image_patch = model(input_sample)
            reconstructed_patches.append(reconstructed_image_patch.unsqueeze(0))

    reconstructed_image = torch.cat(reconstructed_patches, 0).permute(1, 0, 2, 3)
    reconstructed_image = torch.pixel_shuffle(reconstructed_image, patching).squeeze()
    return reconstructed_image

def patched_infer(model, grid, factor):
    batched_grid = grid.permute(2, 0, 1)
    batched_grid = pad_for_shuffling(batched_grid, factor)
    batched_grid = batched_grid.unsqueeze(1)
    batched_grid = torch.pixel_unshuffle(batched_grid, factor)
    batched_grid = batched_grid.permute(1, 2, 3, 0)

    grid_patches = batched_grid.unbind(0)

    return patched_forward(model, grid_patches, factor)