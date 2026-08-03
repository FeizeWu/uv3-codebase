"""Resolution bucketing for ITERABLE parquet (buffer-then-bucket).

transfusion-core's BucketAwareSampler is map-style (needs index+random access), incompatible
with streaming parquet. Here we buffer N decoded samples, group by nearest aspect-ratio bucket,
and yield same-bucket micro-batches (same VAE token count -> no padding waste).
Review (六.5) flagged this adaptation as required.
"""
from __future__ import annotations

import collections
import math

import torch

# FLUX.2: 8x VAE downsample x 2x2 pack = 16x spatial reduction -> token = (H/16)*(W/16).
TOKEN_STRIDE = 16


def default_aspect_buckets(target_pixels: int = 256 * 256) -> dict[float, tuple[int, int]]:
    """A small set of stride-aligned buckets around target_pixels (token count multiples of stride)."""
    side = int(round(math.sqrt(target_pixels)))
    cands = [(side, side), (side * 4 // 3, side * 3 // 4), (side * 3 // 4, side * 4 // 3),
             (side * 3 // 2, side * 2 // 3), (side * 2 // 3, side * 3 // 2)]
    out = {}
    for h, w in cands:
        h = (h // TOKEN_STRIDE) * TOKEN_STRIDE or TOKEN_STRIDE
        w = (w // TOKEN_STRIDE) * TOKEN_STRIDE or TOKEN_STRIDE
        out.setdefault(round(h / w, 3), (h, w))
    return out


def nearest_bucket(h: int, w: int, buckets: dict[float, tuple[int, int]]):
    """Return (h, w) of the nearest aspect-ratio bucket."""
    ar = h / w
    return min(buckets.items(), key=lambda kv: abs(kv[0] - ar))[1]  # (h, w) tuple


class BucketBatcher:
    """Wrap an iterable dataset: buffer -> group by bucket -> yield same-bucket batches."""

    def __init__(self, source_iter, batch_size: int, buf_size: int = 1024, buckets=None):
        self.source = source_iter
        self.bs = batch_size
        self.buf_size = buf_size
        self.buckets = buckets or default_aspect_buckets()

    def __iter__(self):
        buf = collections.defaultdict(list)
        total = 0
        for sample in self.source:
            h, w = sample["pixel_values"].shape[-2:]
            b = nearest_bucket(h, w, self.buckets)
            buf[b].append(sample)
            total += 1
            if total >= self.buf_size:
                yield from self._drain(buf)
                total = 0
        yield from self._drain(buf)

    def _drain(self, buf):
        for b, items in list(buf.items()):
            while len(items) >= self.bs:
                yield items[: self.bs]
                buf[b] = items[self.bs:]
