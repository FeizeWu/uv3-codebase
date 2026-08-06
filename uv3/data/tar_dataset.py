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
import hashlib
import json
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .bucket_sampler import AspectBucket, choose_aspect_bucket
from .transforms import pil_to_tensor


MANIFEST_PATH = "/mnt/oss/uv3-pretrain-manifect/0803-test/manifect.jsonl"
CAPTION_FIELD = "caption_qwen3_7_flash"
META_COLS = ["shard_key", "filename", "offset", "size", "width", "height", "format", CAPTION_FIELD]


@dataclass(frozen=True)
class TarDecodeFailure:
    """Expected malformed-image result; storage and programming errors still raise."""

    reason: str


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


def validate_resume_signature(saved: dict | None, current: dict) -> None:
    if saved != current:
        raise RuntimeError(
            "exact data resume signature changed: "
            f"checkpoint={saved} current={current}"
        )


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
        self.manifest_digest = hashlib.sha256(
            json.dumps(self._entries, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._epoch = 0
        self._resume_shard = 0
        self._resume_row = 0
        self._resume_cursors: dict[int, dict] = {}
        self._worker_rotation = 0

    def set_epoch(self, e):
        self._epoch = e

    def resume_signature(
        self,
        world_size: int,
        workers_per_rank: int,
        *,
        pipeline: dict | None = None,
    ) -> dict:
        signature = {
            "manifest_digest": self.manifest_digest,
            "manifest_entries": len(self._entries),
            "world_size": int(world_size),
            "workers_per_rank": int(workers_per_rank),
            "caption_field": self.caption_field,
            "image_size": int(self.image_size),
            "shuffle": bool(self.shuffle),
            "defer_image_decode": self.defer_image_decode,
            "aspect_buckets": [
                (bucket.name, int(bucket.width), int(bucket.height))
                for bucket in self.aspect_buckets
            ],
        }
        if pipeline is not None:
            signature["pipeline"] = dict(pipeline)
        return signature

    def set_resume(
        self,
        shard_pos: int | None = None,
        row_pos: int | None = None,
        *,
        worker_cursors: dict[int | str, dict] | None = None,
        worker_rotation: int = 0,
    ):
        """Set a direct parquet cursor without replaying earlier samples.

        ``worker_cursors`` records the last descriptor delivered to the main
        process for each logical global worker. Resume opens that worker's
        current parquet directly, locates the saved row group, and continues
        after the saved filtered-row index. The positional arguments retain the
        legacy single-cursor API for old callers.
        """
        if worker_cursors is not None:
            self._resume_cursors = {
                int(worker): dict(cursor) for worker, cursor in worker_cursors.items()
            }
            self._worker_rotation = int(worker_rotation)
            return
        self._resume_shard = int(shard_pos or 0)
        self._resume_row = int(row_pos or 0)

    def __iter__(self):
        worker = get_worker_info()
        physical_wid, nworkers = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        logical_local_wid = (physical_wid + self._worker_rotation) % nworkers
        wid = rank * nworkers + logical_local_wid
        total_workers = world * nworkers

        # Every worker must permute shards identically *before* strided
        # partitioning.  A worker-specific permutation causes overlaps and
        # omissions across ranks even though the slice strides are disjoint.
        shard_indices = partition_shard_indices(
            len(self._entries), 0 if self._resume_cursors else self._resume_shard, self.shuffle,
            self._epoch, wid, total_workers,
        )
        resume_cursor = self._resume_cursors.get(wid)
        if resume_cursor is not None:
            if int(resume_cursor.get("epoch", self._epoch)) != self._epoch:
                raise ValueError(
                    f"worker {wid} resume epoch {resume_cursor.get('epoch')} "
                    f"does not match dataset epoch {self._epoch}"
                )
            resume_shard = int(resume_cursor["shard_pos"])
            try:
                shard_offset = shard_indices.index(resume_shard)
            except ValueError as error:
                raise ValueError(
                    f"worker {wid} does not own resume shard {resume_shard}"
                ) from error
            # Directly open the saved parquet; do not replay prior shards.
            shard_indices = shard_indices[shard_offset:]

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
                # Per-shard seeding makes row-group order directly seekable;
                # it does not depend on opening all preceding parquet files.
                random.Random(
                    5678 + self._epoch * 1_000_003 + wid * 10_007 + si_global
                ).shuffle(row_groups)
            if resume_cursor is not None and si_global == int(resume_cursor["shard_pos"]):
                resume_group = int(resume_cursor["row_group"])
                try:
                    group_offset = row_groups.index(resume_group)
                except ValueError as error:
                    raise ValueError(
                        f"resume row group {resume_group} is absent from shard {si_global}"
                    ) from error
                row_groups = row_groups[group_offset:]

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
                if resume_cursor is not None and (
                    si_global == int(resume_cursor["shard_pos"])
                    and rg == int(resume_cursor["row_group"])
                ):
                    # Cursor is the last descriptor already handed to the
                    # batcher, so resume at the following filtered row.
                    skip = int(resume_cursor["row_pos"]) + 1
                    resume_cursor = None
                elif si_global == self._resume_shard and self._resume_row > 0:
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
                            "row_group": rg,
                            "row_pos": ri,
                            "worker_id": wid,
                            "epoch": self._epoch,
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
                        "row_group": rg,
                        "row_pos": ri,
                        "worker_id": wid,
                        "epoch": self._epoch,
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

    def __call__(self, sample: dict) -> torch.Tensor | TarDecodeFailure:
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
        expected_size = int(sample["size"])
        raw = handle.read(expected_size)
        if len(raw) != expected_size:
            return TarDecodeFailure(
                f"short tar read: expected {expected_size} bytes, got {len(raw)}"
            )
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                return pil_to_tensor(
                    image,
                    (int(sample["image_height"]), int(sample["image_width"])),
                )
        except (OSError, ValueError) as error:
            return TarDecodeFailure(f"{type(error).__name__}: {error}")
