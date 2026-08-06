"""Aspect-ratio buckets shared by streaming datasets and the trainer.

The source image chooses the nearest ratio bucket.  Bucket dimensions keep
approximately ``image_size ** 2`` pixels and are aligned to the full
VAE+MMDiT token stride, so a batch never needs spatial padding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


# FLUX.2: 8x VAE downsample x 2x2 latent patch = 16x image-side/token stride.
TOKEN_STRIDE = 16

# Ratios follow the useful, non-extreme subset of TuVAE's Ideogram profiles.
# At a 256 base resolution, including its 4:1 banner bucket would leave only
# 128 pixels on the short side, which is too destructive for pretraining.
ASPECT_BUCKET_PROFILES: dict[str, tuple[int, int]] = {
    "square": (1, 1),
    "landscape": (3, 2),
    "portrait": (2, 3),
    "widescreen": (16, 9),
    "phone": (9, 16),
}
DEFAULT_ASPECT_BUCKETS = tuple(ASPECT_BUCKET_PROFILES)
ASPECT_BUCKET_ALIASES = {
    "mar_256": DEFAULT_ASPECT_BUCKETS,
    "ideogram5": DEFAULT_ASPECT_BUCKETS,
}


@dataclass(frozen=True)
class AspectBucket:
    name: str
    width: int
    height: int

    @property
    def ratio(self) -> float:
        return self.width / self.height

    @property
    def image_tokens(self) -> int:
        return (self.width // TOKEN_STRIDE) * (self.height // TOKEN_STRIDE)


def normalize_bucket_names(names: str | Iterable[str] | None) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        value = names.strip()
        if not value or value.lower() in {"none", "off", "false"}:
            return ()
        if value in ASPECT_BUCKET_ALIASES:
            return ASPECT_BUCKET_ALIASES[value]
        names = (part.strip() for part in value.split(","))
    result = tuple(name for name in names if name)
    unknown = sorted(set(result) - set(ASPECT_BUCKET_PROFILES))
    if unknown:
        raise ValueError(
            f"unknown aspect buckets {unknown}; expected one of "
            f"{sorted(ASPECT_BUCKET_PROFILES)}"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"aspect bucket names must be unique: {result}")
    return result


def _aligned_bucket_dimensions(
    ratio: float,
    target_pixels: int,
    stride: int,
) -> tuple[int, int]:
    """Round the ideal equal-area rectangle to the nearest aligned dimensions."""
    if target_pixels <= 0 or stride <= 0:
        raise ValueError("target_pixels and stride must be positive")
    ideal_width = math.sqrt(target_pixels * ratio)
    ideal_height = ideal_width / ratio
    width = max(stride, int(round(ideal_width / stride)) * stride)
    height = max(stride, int(round(ideal_height / stride)) * stride)
    return width, height


def build_aspect_buckets(
    image_size: int,
    names: str | Iterable[str] | None = DEFAULT_ASPECT_BUCKETS,
    stride: int = TOKEN_STRIDE,
) -> tuple[AspectBucket, ...]:
    names = normalize_bucket_names(names)
    buckets = []
    for name in names:
        profile_width, profile_height = ASPECT_BUCKET_PROFILES[name]
        width, height = _aligned_bucket_dimensions(
            profile_width / profile_height,
            image_size * image_size,
            stride,
        )
        buckets.append(AspectBucket(name=name, width=width, height=height))
    return tuple(buckets)


def choose_aspect_bucket(
    width: int,
    height: int,
    buckets: Iterable[AspectBucket],
) -> AspectBucket:
    buckets = tuple(buckets)
    if width <= 0 or height <= 0:
        raise ValueError(f"source dimensions must be positive, got {width}x{height}")
    if not buckets:
        raise ValueError("at least one aspect bucket is required")
    source_log_ratio = math.log(width / height)
    return min(
        buckets,
        key=lambda bucket: abs(source_log_ratio - math.log(bucket.ratio)),
    )


def default_aspect_buckets(target_pixels: int = 256 * 256) -> dict[float, tuple[int, int]]:
    """Backward-compatible ``ratio -> (height, width)`` view of default buckets."""
    image_size = int(round(math.sqrt(target_pixels)))
    return {
        round(bucket.height / bucket.width, 6): (bucket.height, bucket.width)
        for bucket in build_aspect_buckets(image_size)
    }


def nearest_bucket(
    h: int,
    w: int,
    buckets: dict[float, tuple[int, int]],
) -> tuple[int, int]:
    """Backward-compatible nearest bucket helper returning ``(height, width)``."""
    log_ratio = math.log(h / w)
    return min(buckets.items(), key=lambda item: abs(math.log(item[0]) - log_ratio))[1]
