"""FLUX.2 VAE wrapper with external BatchNorm latent normalization.

Follows UniWorld third_party/DiT/flux2_vae.py + official diffusers pipeline_flux2:
the FLUX.2 VAE has NO scale_factor/shift_factor; instead a BatchNorm (vae.bn) over the
128 packed channels is applied EXTERNALLY (patchify -> (x-mean)/std -> unpatchify).
This module does NOT use raw latents — encode_images/decode_latents always apply BN.
"""
from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*4, H/2, W/2) via 2x2 vec-pack."""
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError(f"latent dims must be even, got {h}x{w}")
    return (
        latents.view(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * 4, h // 2, w // 2)
    )


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C*4, H, W) -> (B, C, H*2, W*2)."""
    b, c4, h, w = latents.shape
    return (
        latents.view(b, c4 // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c4 // 4, h * 2, w * 2)
    )


def match_input_channels(vae, images: torch.Tensor) -> torch.Tensor:
    """No-op for 3-channel RGB (FLUX.2 in_channels=3); pads alpha for a future 4ch VAE."""
    in_ch = int(getattr(vae.config, "in_channels", images.shape[1]))
    if images.shape[1] == in_ch:
        return images
    if images.shape[1] == 3 and in_ch == 4:
        alpha = torch.ones(
            images.shape[0], 1, images.shape[2], images.shape[3],
            device=images.device, dtype=images.dtype,
        )
        return torch.cat([images, alpha], dim=1)
    raise ValueError(f"cannot encode {images.shape[1]}-ch images with {in_ch}-ch VAE")


class Flux2VAE(torch.nn.Module):
    """Frozen FLUX.2 VAE with BatchNorm external latent normalization."""

    def __init__(self, vae: AutoencoderKLFlux2):
        super().__init__()
        self.vae = vae
        self.requires_grad_(False)
        self.eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))

    @classmethod
    def from_pretrained(cls, path: str, dtype=torch.bfloat16, force_upcast: bool | None = None):
        p = Path(path)
        subfolder = None if (p / "config.json").is_file() else "vae"
        vae = AutoencoderKLFlux2.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype)
        if force_upcast is not None:
            vae.config.force_upcast = force_upcast  # False -> bf16 VAE (faster, less mem)
        return cls(vae).to(dtype=dtype)

    @property
    def latent_channels(self) -> int:
        return int(self.vae.config.latent_channels)

    @property
    def dtype(self):
        return next(self.vae.parameters()).dtype

    def scale_factor(self, image_size: int) -> int:
        """image-side downsample factor = 2**(len(block_out_channels)-1)."""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def latent_spec(self, image_size: int):
        """(latent_channels, latent_spatial) for a square image."""
        sf = self.scale_factor(image_size)
        if image_size % sf:
            raise ValueError(f"image_size {image_size} not divisible by VAE scale {sf}")
        return self.latent_channels, image_size // sf

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """RGB images -> normalized latents (B, C, H/sf, W/sf)."""
        images = match_input_channels(self.vae, images)
        raw = self.vae.encode(images).latent_dist.sample()
        # BN normalization applied on the PACKED (128-ch) space, matching official pipeline.
        patched = patchify_latents(raw)
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(patched)
        var = self.vae.bn.running_var.view(1, -1, 1, 1).to(patched)
        std = (var + self.eps).sqrt()
        return unpatchify_latents((patched - mean) / std)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """normalized latents -> RGB images."""
        patched = patchify_latents(latents)
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(patched)
        var = self.vae.bn.running_var.view(1, -1, 1, 1).to(patched)
        std = (var + self.eps).sqrt()
        raw = unpatchify_latents(patched * std + mean)
        return self.vae.decode(raw).sample
