"""TarMetadataDataset: stream images from tar archives via manifest + metadata parquet.

Reads: manifest jsonl → metadata parquet (shard_key/filename/offset/size/caption/width/height/format)
       → tar seek(offset) read(size) → decode.
Filters: caption non-null (~11% drop).
Shuffle: block-level (inter-shard), in-block by offset (tar-friendly sequential read).
Resume: data_status=(shard_pos, row_pos) row-level, stored in ckpt.
Bucketing: aspect hard bucket + caption-length binning within bucket (same bin = same batch).
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .transforms import center_crop_resize


MANIFEST_PATH = "/mnt/oss/uv3-pretrain-manifect/0803-test/manifect.jsonl"
CAPTION_FIELD = "caption_qwen3_7_flash"
META_COLS = ["shard_key", "filename", "offset", "size", "width", "height", "format", CAPTION_FIELD]


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

    def __init__(self, manifest_path: str = MANIFEST_PATH, image_size: int = 256,
                 caption_field: str = CAPTION_FIELD, shuffle: bool = True):
        super().__init__()
        self.manifest_path = manifest_path
        self.image_size = image_size
        self.caption_field = caption_field
        self.shuffle = shuffle
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
        rng = random.Random(1234 + self._epoch + wid)

        # shard order: inter-shard shuffle (skipping resumed shards)
        shard_indices = list(range(self._resume_shard, len(self._entries)))
        if self.shuffle and self._resume_shard == 0:
            shard_indices = list(range(len(self._entries)))
            rng.shuffle(shard_indices)
        # partition across workers
        shard_indices = shard_indices[wid::(world * nworkers)]

        # cache tar file handles by path
        tar_cache = {}

        for si_local, si_global in enumerate(shard_indices):
            image_tar, metadata_parquet = self._entries[si_global]
            pf = pq.ParquetFile(metadata_parquet)
            row_groups = list(range(pf.num_row_groups))
            if self.shuffle:
                rng.shuffle(row_groups)

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
                    # read image from tar (byte-range via seek)
                    try:
                        if image_tar not in tar_cache:
                            tar_cache[image_tar] = open(image_tar, "rb")
                        fh = tar_cache[image_tar]
                        fh.seek(row["offset"])
                        raw = fh.read(row["size"])
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        img = center_crop_resize(img, self.image_size)
                        arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
                        arr = arr.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)
                    except Exception:
                        continue

                    caption = row[self.caption_field]
                    yield {
                        "pixel_values": arr,
                        "text": caption,
                        "shard_pos": si_global,
                        "row_pos": ri,
                    }

        for fh in tar_cache.values():
            try:
                fh.close()
            except Exception:
                pass
