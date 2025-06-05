from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def resize_by_factor(tensor, scale, interpolation_mode):
    return resize(
        tensor, (tensor.size(-2) // scale, tensor.size(-1) // scale), 
        antialias=True,
        interpolation = interpolation_mode
    )


def downsample_grid(grid, scale):
    print(grid.shape)
    grid = grid.movedim(-1, 0)
    print(grid.shape)
    resized = resize_by_factor(grid, scale, interpolation_mode = InterpolationMode.NEAREST)
    print(resized.shape)
    resized = resized.movedim(0, -1)
    print(resized.shape)
    return resized

def downsample_image(image, scale):
    return resize_by_factor(image, scale, InterpolationMode.BILINEAR)

def build_downsample_case(grid, image, scale):
    downsampled_grid = downsample_grid(grid, scale)
    downsampled_image = downsample_image(image, scale)

    return [(downsampled_grid, downsampled_image)]
