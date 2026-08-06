from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from uv3.data.bucket_sampler import AspectBucket
from uv3.data.tar_dataset import TarMetadataDataset, validate_resume_signature
from uv3.train.fsdp2_trainer import DataPrefetcher, OnlineJointBucketBatcher


def _write_manifest(tmp_path: Path, shards: int = 3) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    entries = []
    for shard in range(shards):
        parquet = tmp_path / f"meta-{shard}.parquet"
        rows = 4
        table = pa.table({
            "shard_key": [f"s{shard}"] * rows,
            "filename": [f"{row}.png" for row in range(rows)],
            "offset": [row * 10 for row in range(rows)],
            "size": [10] * rows,
            "width": [32] * rows,
            "height": [32] * rows,
            "format": ["png"] * rows,
            "caption_qwen3_7_flash": [f"s{shard}-r{row}" for row in range(rows)],
        })
        pq.write_table(table, parquet, row_group_size=2)
        entries.append({"image_tar": str(tmp_path / f"image-{shard}.tar"),
                        "metadata_parquet": str(parquet)})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in entries))
    return manifest


def test_tar_resume_opens_saved_parquet_directly_and_skips_only_inside_row_group(
    tmp_path, monkeypatch,
):
    manifest = _write_manifest(tmp_path)
    dataset = TarMetadataDataset(
        str(manifest), shuffle=False, defer_image_decode=True,
    )
    iterator = iter(dataset)
    consumed = [next(iterator) for _ in range(7)]
    cursor_sample = consumed[-1]
    cursor = {
        key: cursor_sample[key]
        for key in ("epoch", "shard_pos", "row_group", "row_pos")
    }

    import uv3.data.tar_dataset as tar_dataset_module
    opened = []
    original = tar_dataset_module.pq.ParquetFile

    def tracking_parquet(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(tar_dataset_module.pq, "ParquetFile", tracking_parquet)
    resumed = TarMetadataDataset(
        str(manifest), shuffle=False, defer_image_decode=True,
    )
    resumed.set_resume(worker_cursors={0: cursor})
    remaining = list(resumed)

    assert [row["text"] for row in remaining] == [
        "s1-r3", "s2-r0", "s2-r1", "s2-r2", "s2-r3",
    ]
    assert opened == ["meta-1.parquet", "meta-2.parquet"]


def test_resume_signature_rejects_manifest_or_topology_changes(tmp_path):
    manifest = _write_manifest(tmp_path)
    dataset = TarMetadataDataset(str(manifest), defer_image_decode=True)
    signature = dataset.resume_signature(
        world_size=4,
        workers_per_rank=4,
        pipeline={"batch_size_per_gpu": 12, "tokenizer": "qwen"},
    )
    validate_resume_signature(signature, dict(signature))

    changed_workers = dict(signature, workers_per_rank=2)
    with pytest.raises(RuntimeError, match="resume signature changed"):
        validate_resume_signature(signature, changed_workers)

    changed_manifest = dict(signature, manifest_digest="different")
    with pytest.raises(RuntimeError, match="resume signature changed"):
        validate_resume_signature(signature, changed_manifest)

    changed_pipeline = dataset.resume_signature(
        world_size=4,
        workers_per_rank=4,
        pipeline={"batch_size_per_gpu": 8, "tokenizer": "qwen"},
    )
    with pytest.raises(RuntimeError, match="resume signature changed"):
        validate_resume_signature(signature, changed_pipeline)


def test_four_worker_stream_resume_preserves_next_descriptor_order(tmp_path):
    manifest = _write_manifest(tmp_path, shards=20)
    original_dataset = TarMetadataDataset(
        str(manifest), shuffle=True, defer_image_decode=True,
    )
    original_iterator = iter(DataLoader(
        original_dataset, batch_size=None, num_workers=4,
    ))
    cursors = {}
    last_worker = None
    for _ in range(29):
        sample = next(original_iterator)
        last_worker = int(sample["worker_id"])
        cursors[last_worker] = {
            key: int(sample[key])
            for key in ("epoch", "shard_pos", "row_group", "row_pos")
        }
    expected = [next(original_iterator)["text"] for _ in range(24)]

    resumed_dataset = TarMetadataDataset(
        str(manifest), shuffle=True, defer_image_decode=True,
    )
    resumed_dataset.set_resume(
        worker_cursors=cursors,
        worker_rotation=(last_worker + 1) % 4,
    )
    resumed_iterator = iter(DataLoader(
        resumed_dataset, batch_size=None, num_workers=4,
    ))
    actual = [next(resumed_iterator)["text"] for _ in range(24)]

    assert actual == expected


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [list(range(int(text.split(":", 1)[0]))) for text in texts]}


def _samples(count=80):
    rows = []
    for index in range(count):
        length = 3 if index % 3 else 7
        rows.append({
            "text": f"{length}:sample-{index}",
            "resolution_bucket": "square",
            "image_height": 32,
            "image_width": 32,
            "worker_id": index % 4,
            "epoch": 0,
            "shard_pos": index // 8,
            "row_group": (index // 2) % 4,
            "row_pos": index % 2,
            "value": index,
        })
    return rows


def _batcher(samples):
    return OnlineJointBucketBatcher(
        samples,
        _Tokenizer(),
        batch_size=2,
        text_buckets=(4, 8),
        text_weights=(1, 1),
        resolution_buckets=(AspectBucket("square", 32, 32),),
        tokenize_batch_size=5,
        max_buffer_samples=100,
        decode_workers=1,
        decode_prefetch_batches=2,
        decode_fn=lambda sample: torch.full((3, 32, 32), float(sample["value"])),
    )


def _batch_identity(batch):
    return batch["text"], batch["pixel_values"][:, 0, 0, 0].tolist()


def test_online_joint_queue_and_pending_specs_resume_exactly():
    samples = _samples()
    uninterrupted = iter(_batcher(samples))
    expected = [_batch_identity(next(uninterrupted)) for _ in range(10)]
    uninterrupted.close()

    original = _batcher(samples)
    iterator = iter(original)
    prefix = [_batch_identity(next(iterator)) for _ in range(3)]
    state = original.state_dict()
    source_offset = original.source_samples
    iterator.close()

    resumed = _batcher(samples[source_offset:])
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    suffix = [_batch_identity(next(resumed_iterator)) for _ in range(7)]
    resumed_iterator.close()

    assert prefix + suffix == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefetch stream required")
def test_data_prefetcher_restores_materialized_next_batch_without_consuming_source():
    def batch(label, value):
        return {
            "pixel_values": torch.full((1, 3, 4, 4), value),
            "text": [label],
        }

    first = batch("first", 1.0)
    prefetcher = DataPrefetcher([first], torch.device("cuda", 0), torch.bfloat16)
    state = prefetcher.state_dict()
    resumed = DataPrefetcher(
        [batch("second", 2.0)],
        torch.device("cuda", 0),
        torch.bfloat16,
        resume_state=state,
    )

    restored_first = resumed.next()
    restored_second = resumed.next()
    assert restored_first["text"] == ["first"]
    assert restored_first["pixel_values"][0, 0, 0, 0].item() == 1.0
    assert restored_second["text"] == ["second"]
    assert restored_second["pixel_values"][0, 0, 0, 0].item() == 2.0
