"""Parquet streaming dataset (pyarrow row-group), reusing UniWorld ImageNetParquetDataset pattern.

ImageNet parquet columns: 'image' (bytes), 'label' (int). Caption = class-name template.
Supports overfit_n: take the first N images and repeat (deterministic overfit set).
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .bucket_sampler import AspectBucket, choose_aspect_bucket
from .transforms import pil_to_tensor

# ImageNet 1k class names are huge; for overfit/efficiency we use a generic prompt keyed by label.
_PROMPT = "a photo of class {label}"


class ParquetImageDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        parquet_glob: str = "data/train-*.parquet",
        image_size: int = 256,
        overfit_n: int | None = None,
        image_field: str = "image",
        label_field: str = "label",
        aspect_buckets: tuple[AspectBucket, ...] = (),
    ):
        super().__init__()
        data_dir = Path(root)
        self.shards = sorted((data_dir if (data_dir / parquet_glob.split("/")[0]).is_dir() else data_dir).glob(parquet_glob.split("/", 1)[-1] if "/" in parquet_glob else parquet_glob))
        if not self.shards:
            # fallback: glob directly under root
            self.shards = sorted(Path(root).rglob(parquet_glob.split("/")[-1]))
        self.image_size = image_size
        self.overfit_n = overfit_n
        self.image_field = image_field
        self.label_field = label_field
        self.aspect_buckets = tuple(aspect_buckets)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _read_rows(self, path, row_groups):
        pf = pq.ParquetFile(path)
        out = []
        for rg in row_groups:
            out.extend(pf.read_row_group(rg, columns=[self.image_field, self.label_field]).to_pylist())
        return out, pf

    def __iter__(self):
        worker = get_worker_info()
        wid, nworkers = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        wid = rank * nworkers + wid
        rng = random.Random(1234 + self._epoch + wid)
        shards = list(self.shards[wid::(world * nworkers)])
        rng.shuffle(shards)

        if self.overfit_n is not None:
            # deterministic overfit: read first shard, take first overfit_n rows, repeat forever
            if not shards:
                return
            pf = pq.ParquetFile(shards[0])
            rows = pf.read_row_group(0, columns=[self.image_field, self.label_field]).to_pylist()[: self.overfit_n]
            while True:
                for row in rows:
                    yield self._make(row)
        else:
            for path in shards:
                pf = pq.ParquetFile(path)
                rgs = list(range(pf.num_row_groups))
                rng.shuffle(rgs)
                for rg in rgs:
                    rows = pf.read_row_group(rg, columns=[self.image_field, self.label_field]).to_pylist()
                    rng.shuffle(rows)
                    for row in rows:
                        yield self._make(row)

    def _make(self, row):
        img = row[self.image_field]
        if isinstance(img, dict):
            img = img.get("bytes")
        label = int(row[self.label_field])
        image = Image.open(io.BytesIO(img)).convert("RGB")
        if self.aspect_buckets:
            bucket = choose_aspect_bucket(image.width, image.height, self.aspect_buckets)
        else:
            bucket = AspectBucket(
                name="square",
                width=self.image_size,
                height=self.image_size,
            )
        return {
            "pixel_values": pil_to_tensor(image, (bucket.height, bucket.width)),
            "text": _PROMPT.format(label=label),
            "label": label,
            "resolution_bucket": bucket.name,
            "image_height": bucket.height,
            "image_width": bucket.width,
        }
