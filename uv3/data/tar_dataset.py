"""TarMetadataDataset: stream images from tar archives via manifest + metadata parquet.

Reads: manifest jsonl → metadata parquet (shard_key/filename/offset/size/caption/width/height/format)
       → tar seek(offset) read(size) → decode.
Filters: caption non-null (~11% drop).
Shuffle: block-level (inter-shard), in-block by offset (tar-friendly sequential read).
Resume: data_status=(shard_pos, row_pos) row-level, stored in ckpt.
Bucketing: optionally emit compact descriptors for online joint text/aspect scheduling;
           selected descriptors are decoded later in a bounded thread pool.
"""
from __future__ import annotations

import io
import json
import random
import threading
from collections import OrderedDict

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .bucket_sampler import AspectBucket, choose_aspect_bucket
from .transforms import pil_to_tensor


MANIFEST_PATH = "/mnt/oss/uv3-pretrain-manifect/0803-test/manifect.jsonl"
CAPTION_FIELD = "caption_qwen3_7_flash"
META_COLS = ["shard_key", "filename", "offset", "size", "width", "height", "format", CAPTION_FIELD]


def partition_shard_indices(
    entry_count: int,
    resume_shard: int,
    shuffle: bool,
    epoch: int,
    worker_index: int,
    total_workers: int,
) -> list[int]:
    """Return one worker's disjoint slice of a shared deterministic order."""
    shard_indices = list(range(resume_shard, entry_count))
    if shuffle and resume_shard == 0:
        random.Random(1234 + epoch).shuffle(shard_indices)
    return shard_indices[worker_index::total_workers]


def _load_manifest(manifest_path: str):
    """Return list of (image_tar, metadata_parquet) from manifest jsonl."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            d = json.loads(line.strip())
            entries.append((d["image_tar"], d["metadata_parquet"]))
    return entries


class TarMetadataDataset(IterableDataset):
    """Stream images from tar via metadata parquet, with caption."""

    def __init__(
        self,
        manifest_path: str = MANIFEST_PATH,
        image_size: int = 256,
        caption_field: str = CAPTION_FIELD,
        shuffle: bool = True,
        aspect_buckets: tuple[AspectBucket, ...] = (),
        defer_image_decode: bool = False,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.image_size = image_size
        self.caption_field = caption_field
        self.shuffle = shuffle
        self.aspect_buckets = tuple(aspect_buckets)
        self.defer_image_decode = bool(defer_image_decode)
        self._entries = _load_manifest(manifest_path)
        self._epoch = 0
        self._resume_shard = 0
        self._resume_row = 0

    def set_epoch(self, e):
        self._epoch = e

    def set_resume(self, shard_pos: int, row_pos: int):
        self._resume_shard = shard_pos
        self._resume_row = row_pos

    def __iter__(self):
        worker = get_worker_info()
        wid, nworkers = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        wid = rank * nworkers + wid
        total_workers = world * nworkers

        # Every worker must permute shards identically *before* strided
        # partitioning.  A worker-specific permutation causes overlaps and
        # omissions across ranks even though the slice strides are disjoint.
        shard_indices = partition_shard_indices(
            len(self._entries), self._resume_shard, self.shuffle,
            self._epoch, wid, total_workers,
        )
        # Row-group order may differ per worker after shard ownership is fixed.
        worker_rng = random.Random(5678 + self._epoch * total_workers + wid)

        # Bound open files well below the common RLIMIT_NOFILE=1024.  The
        # manifest contains tens of thousands of tar paths, so an unbounded
        # cache eventually makes all later reads fail with EMFILE.
        tar_cache = OrderedDict()
        max_open_tars = 64

        for si_local, si_global in enumerate(shard_indices):
            image_tar, metadata_parquet = self._entries[si_global]
            pf = pq.ParquetFile(metadata_parquet)
            row_groups = list(range(pf.num_row_groups))
            if self.shuffle:
                worker_rng.shuffle(row_groups)

            for rg in row_groups:
                try:
                    rows = pf.read_row_group(rg, columns=META_COLS).to_pylist()
                except Exception:
                    continue

                # filter caption non-null + sort by offset (tar sequential read)
                rows = [r for r in rows if r.get(self.caption_field)]
                rows.sort(key=lambda r: r["offset"])

                # resume skip (only for first shard)
                skip = 0
                if si_global == self._resume_shard and self._resume_row > 0:
                    skip = self._resume_row
                    self._resume_row = 0  # only skip once

                for ri, row in enumerate(rows):
                    if ri < skip:
                        continue
                    source_width = int(row.get("width") or 0)
                    source_height = int(row.get("height") or 0)
                    if self.defer_image_decode and (source_width <= 0 or source_height <= 0):
                        continue
                    if self.aspect_buckets and source_width > 0 and source_height > 0:
                        bucket = choose_aspect_bucket(
                            source_width, source_height, self.aspect_buckets,
                        )
                    else:
                        bucket = AspectBucket(
                            name="square",
                            width=self.image_size,
                            height=self.image_size,
                        )
                    caption = row[self.caption_field]
                    if self.defer_image_decode:
                        # Keep the online bucket queue compact: no decoded tensor,
                        # shared-memory storage, or open tar descriptor is retained.
                        yield {
                            "text": caption,
                            "resolution_bucket": bucket.name,
                            "image_height": bucket.height,
                            "image_width": bucket.width,
                            "image_tar": image_tar,
                            "offset": int(row["offset"]),
                            "size": int(row["size"]),
                            "shard_pos": si_global,
                            "row_pos": ri,
                        }
                        continue
                    # read image from tar (byte-range via seek)
                    try:
                        if image_tar not in tar_cache:
                            if len(tar_cache) >= max_open_tars:
                                _, old_handle = tar_cache.popitem(last=False)
                                old_handle.close()
                            tar_cache[image_tar] = open(image_tar, "rb")
                        else:
                            tar_cache.move_to_end(image_tar)
                        fh = tar_cache[image_tar]
                        fh.seek(row["offset"])
                        raw = fh.read(row["size"])
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        source_width = source_width or img.width
                        source_height = source_height or img.height
                        if self.aspect_buckets and (source_width, source_height) != (0, 0):
                            bucket = choose_aspect_bucket(
                                source_width, source_height, self.aspect_buckets,
                            )
                        arr = pil_to_tensor(img, (bucket.height, bucket.width))
                    except Exception:
                        continue

                    yield {
                        "pixel_values": arr,
                        "text": caption,
                        "resolution_bucket": bucket.name,
                        "image_height": bucket.height,
                        "image_width": bucket.width,
                        "shard_pos": si_global,
                        "row_pos": ri,
                    }

        for fh in tar_cache.values():
            try:
                fh.close()
            except Exception:
                pass


class TarDescriptorDecoder:
    """Thread-safe callable that decodes selected descriptors with per-thread LRU handles."""

    def __init__(self, max_open_tars_per_thread: int = 16):
        self.max_open_tars_per_thread = max(1, int(max_open_tars_per_thread))
        self._local = threading.local()

    def _cache(self) -> OrderedDict:
        cache = getattr(self._local, "tar_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._local.tar_cache = cache
        return cache

    def __call__(self, sample: dict) -> torch.Tensor:
        cache = self._cache()
        image_tar = sample["image_tar"]
        if image_tar not in cache:
            if len(cache) >= self.max_open_tars_per_thread:
                _, old_handle = cache.popitem(last=False)
                old_handle.close()
            cache[image_tar] = open(image_tar, "rb")
        else:
            cache.move_to_end(image_tar)
        handle = cache[image_tar]
        handle.seek(int(sample["offset"]))
        raw = handle.read(int(sample["size"]))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return pil_to_tensor(
            image,
            (int(sample["image_height"]), int(sample["image_width"])),
        )
