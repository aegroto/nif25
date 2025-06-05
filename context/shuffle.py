import torch
from torchvision.transforms.functional import crop

from phases.infer import pad_for_shuffling

def crop_tensor(input, cropping):
    blocks = list()

    width = input.size(-1)
    height = input.size(-2)

    block_width = width // cropping
    block_height = height // cropping
    for y_offset in range(0, height, block_height):
        for x_offset in range(0, width, block_width):
            block = crop(input, y_offset, x_offset, block_height, block_width)
            blocks.append(block)
    
    blocks_tensor = torch.stack(blocks).to(input.device)

    return blocks_tensor

def shuffle_grid(grid, factor):
    batched_grid = grid.permute(2, 0, 1)
    batched_grid = pad_for_shuffling(batched_grid, factor)
    batched_grid = batched_grid.unsqueeze(1)
    batched_grid = torch.pixel_unshuffle(batched_grid, factor)
    batched_grid = batched_grid.permute(1, 2, 3, 0)
    return batched_grid.unbind(0)

def shuffle_image(image, factor):
    batched_image = image.unsqueeze(1)
    batched_image = pad_for_shuffling(batched_image, factor)
    batched_image = torch.pixel_unshuffle(batched_image, factor)
    batched_image = batched_image.permute(1, 0, 2, 3)
    return batched_image.unbind(0)

def crop_grid(grid, factor):
    batched_grid = grid.permute(2, 0, 1)
    batched_grid = pad_for_shuffling(batched_grid, factor)
    batched_grid = crop_tensor(batched_grid, factor)
    batched_grid = batched_grid.permute(0, 2, 3, 1)
    return batched_grid.unbind(0)

def crop_image(image, factor):
    batched_image = pad_for_shuffling(image, factor)
    batched_image = crop_tensor(batched_image, factor)
    return batched_image.unbind(0)

def build_shuffle_case(grid, image, scale, accumulation=1):
    grids = shuffle_grid(grid, scale)
    planes = shuffle_image(image, scale)
    training_pairs = list(zip(grids, planes))

    training_batches = list()
    for (pair_grid, pair_plane) in training_pairs:
        batch_grids = crop_grid(pair_grid, accumulation)
        batch_planes = crop_image(pair_plane, accumulation)
        batch = list(zip(batch_grids, batch_planes))
        training_batches.append(batch)

    return training_batches
