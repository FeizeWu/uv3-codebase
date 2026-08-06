"""Image transforms. RGB-only for now (FLUX.2 VAE is 3ch; alpha deferred)."""
from __future__ import annotations

import io

import torch
from PIL import Image


def _target_hw(image_size: int | tuple[int, int], width: int | None = None) -> tuple[int, int]:
    if isinstance(image_size, tuple):
        if width is not None:
            raise ValueError("width must be omitted when image_size is a (height, width) tuple")
        height, width = image_size
    else:
        height = image_size
        width = image_size if width is None else width
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"target dimensions must be positive, got {height}x{width}")
    return height, width


def center_crop_resize(
    image: Image.Image,
    image_size: int | tuple[int, int],
    width: int | None = None,
) -> Image.Image:
    """Center-crop to the target aspect ratio and resize to ``(height, width)``."""
    target_height, target_width = _target_hw(image_size, width)
    while image.width >= 2 * target_width and image.height >= 2 * target_height:
        image = image.resize(tuple(s // 2 for s in image.size), resample=Image.Resampling.BOX)
    scale = max(target_width / image.width, target_height / image.height)
    image = image.resize(tuple(round(s * scale) for s in image.size), resample=Image.Resampling.BICUBIC)
    left = (image.width - target_width) // 2
    top = (image.height - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def pil_to_tensor(
    image: Image.Image,
    image_size: int | tuple[int, int],
    width: int | None = None,
) -> torch.Tensor:
    """PIL RGB -> (3, H, W) float in [-1, 1]."""
    image = image.convert("RGB")
    target_height, target_width = _target_hw(image_size, width)
    image = center_crop_resize(image, (target_height, target_width))
    arr = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return arr.reshape(target_height, target_width, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)


def decode_image(
    image_bytes: bytes,
    image_size: int | tuple[int, int],
    width: int | None = None,
) -> torch.Tensor:
    return pil_to_tensor(Image.open(io.BytesIO(image_bytes)), image_size, width)
