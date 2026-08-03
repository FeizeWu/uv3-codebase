"""Image transforms. RGB-only for now (FLUX.2 VAE is 3ch; alpha deferred)."""
from __future__ import annotations

import io

import torch
from PIL import Image


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop to image_size (matches UniWorld dit_center_crop)."""
    while min(image.size) >= 2 * image_size:
        image = image.resize(tuple(s // 2 for s in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(image.size)
    image = image.resize(tuple(round(s * scale) for s in image.size), resample=Image.Resampling.BICUBIC)
    left = (image.width - image_size) // 2
    top = (image.height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """PIL RGB -> (3, H, W) float in [-1, 1]."""
    image = image.convert("RGB")
    image = center_crop_resize(image, image_size)
    arr = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return arr.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)


def decode_image(image_bytes: bytes, image_size: int) -> torch.Tensor:
    return pil_to_tensor(Image.open(io.BytesIO(image_bytes)), image_size)
