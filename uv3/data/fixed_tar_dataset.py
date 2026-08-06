"""Infinite, distributed repetition of an exact fixed set of tar-backed samples."""
from __future__ import annotations

import io
import json
import random

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .bucket_sampler import AspectBucket, choose_aspect_bucket
from .transforms import pil_to_tensor


class FixedTarSampleDataset(IterableDataset):
    def __init__(
        self,
        cases_path: str,
        image_size: int = 256,
        shuffle: bool = True,
        aspect_buckets: tuple[AspectBucket, ...] = (),
    ):
        super().__init__()
        self.image_size = image_size
        self.shuffle = shuffle
        self.aspect_buckets = tuple(aspect_buckets)
        with open(cases_path, encoding="utf-8") as file:
            self.cases = [json.loads(line) for line in file if line.strip()]
        if not self.cases:
            raise ValueError(f"fixed overfit set is empty: {cases_path}")

    def set_epoch(self, epoch: int) -> None:
        # Kept for the trainer's common dataset interface. The iterator advances
        # its own deterministic cycle counter because it is intentionally infinite.
        self.initial_epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # A tiny fixed set must retain every text-length bucket on every rank.
        # Shard only between local DataLoader workers; ranks consume different
        # deterministic shuffles of the same exact overfit set.
        global_worker = rank * num_workers + worker_id
        total_workers = world * num_workers
        indices = list(range(worker_id, len(self.cases), num_workers))
        if not indices:
            raise RuntimeError(
                f"fixed set has {len(self.cases)} samples but {total_workers} workers"
            )
        handles = {}
        cycle = getattr(self, "initial_epoch", 0)
        try:
            while True:
                order = list(indices)
                if self.shuffle:
                    random.Random(20260805 + cycle * total_workers + global_worker).shuffle(order)
                for index in order:
                    case = self.cases[index]
                    tar_path = case["image_tar"]
                    try:
                        if tar_path not in handles:
                            handles[tar_path] = open(tar_path, "rb")
                        file = handles[tar_path]
                        file.seek(case["offset"])
                        raw = file.read(case["size"])
                        image = Image.open(io.BytesIO(raw)).convert("RGB")
                        if self.aspect_buckets:
                            bucket = choose_aspect_bucket(
                                int(case.get("width") or image.width),
                                int(case.get("height") or image.height),
                                self.aspect_buckets,
                            )
                        else:
                            bucket = AspectBucket(
                                name="square",
                                width=self.image_size,
                                height=self.image_size,
                            )
                        pixels = pil_to_tensor(image, (bucket.height, bucket.width))
                    except Exception:
                        continue
                    yield {
                        "pixel_values": pixels,
                        "text": case["caption"],
                        "case_id": case["case_id"],
                        "resolution_bucket": bucket.name,
                        "image_height": bucket.height,
                        "image_width": bucket.width,
                    }
                cycle += 1
        finally:
            for file in handles.values():
                file.close()
