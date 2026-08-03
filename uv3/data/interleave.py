"""t2i + it2i interleaving for joint pretraining.

Reuses only the edit-pair DATA SEMANTICS from transfusion-core interleave_datasets
(source image + instruction + target image); NOT the transfusion single-sequence packing
(our MMDiT uses separate [txt, ref, img] streams via num_ref_tokens). Review (三 ⚠) flagged
this distinction. it2i pairs come from a sidecar parquet (image, instruction, target_image);
a fraction it2i_mix of batches are edit pairs (ref=source latent), rest are t2i (ref=None).
"""
from __future__ import annotations

import random

import torch

from .parquet_dataset import ParquetImageDataset


class InterleaveT2iIt2i:
    """Stream t2i (from t2i parquet) + it2i edit pairs (from it2i parquet) by mix fraction."""

    def __init__(self, t2i_root, it2i_root, image_size, it2i_mix=0.0, it2i_field="instruction", **t2i_kwargs):
        self.t2i = ParquetImageDataset(root=t2i_root, image_size=image_size, **t2i_kwargs)
        self.it2i = ParquetImageDataset(root=it2i_root, image_size=image_size, **t2i_kwargs) if it2i_root else None
        self.it2i_mix = it2i_mix
        self.it2i_field = it2i_field
        self.image_size = image_size

    def set_epoch(self, e):
        self.t2i.set_epoch(e)
        if self.it2i:
            self.it2i.set_epoch(e)

    def __iter__(self):
        if self.it2i is None or self.it2i_mix <= 0:
            for s in self.t2i:
                yield {"pixel_values": s["pixel_values"], "text": s["text"], "ref": None}
            return
        it_t2i = iter(self.t2i)
        it_it2i = iter(self.it2i)
        rng = random.Random(1234)
        while True:
            try:
                if rng.random() < self.it2i_mix:
                    s = next(it_it2i)
                    # it2i: source image as ref, instruction as text
                    yield {"pixel_values": s["pixel_values"], "text": s.get(self.it2i_field, s["text"]),
                           "ref": s.get("source_image", s["pixel_values"])}
                else:
                    s = next(it_t2i)
                    yield {"pixel_values": s["pixel_values"], "text": s["text"], "ref": None}
            except StopIteration:
                return
