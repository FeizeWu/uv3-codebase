"""Two-or-more-rank runtime test for Gloo bucket control and local resolutions."""
from __future__ import annotations

import os

import torch

from uv3.data.bucket_sampler import AspectBucket
from uv3.data.dynamic_bucket_scheduler import DistributedBucketController
from uv3.train.fsdp2_trainer import OnlineJointBucketBatcher


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [list(range(int(text.split(":", 1)[0]))) for text in texts]}


def _samples(rank: int, count: int = 400):
    resolution = "square" if rank % 2 == 0 else "landscape"
    width = 32 if resolution == "square" else 48
    rows = []
    for index in range(count):
        length = (2, 4, 6, 8, 10)[index % 5] if rank == 0 else 10
        rows.append({
            "id": rank * 10_000 + index,
            "text": f"{length}:rank-{rank}-sample-{index}",
            "resolution_bucket": resolution,
            "image_height": 32,
            "image_width": width,
            "worker_id": rank,
            "epoch": 0,
            "shard_pos": rank,
            "row_group": index // 100,
            "row_pos": index % 100,
        })
    return rows


def main() -> None:
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError("distributed bucket runtime requires at least two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    group = dist.new_group(backend="gloo")
    controller = DistributedBucketController(group, bucket_count=5)

    local_mask = 0b11111 if rank == 0 else 0b11100
    result = controller.synchronize(
        local_mask,
        has_completed_window=True,
        lookahead_exhausted=rank != 0,
    )
    assert result.common_ready_mask == 0b11100
    assert result.all_have_completed_window
    assert result.any_lookahead_exhausted

    local_counts = (8, 6, 3, 2, 1) if rank == 0 else (2, 3, 4, 5, 6)
    weights, event, _ = controller.update_weights(
        local_counts,
        old_weights=(20, 18, 18, 14, 30),
        schedule_version=0,
        safety_margin=0.02,
    )
    assert weights == (8, 15, 20, 25, 32)
    assert event["schedule_version"] == 1
    assert event["samples_per_rank_min"] == event["samples_per_rank_max"] == 20

    batcher = OnlineJointBucketBatcher(
        _samples(rank),
        _Tokenizer(),
        batch_size=2,
        text_buckets=(2, 4, 6, 8, 10),
        text_weights=(20, 18, 18, 14, 30),
        resolution_buckets=(
            AspectBucket("square", 32, 32),
            AspectBucket("landscape", 48, 32),
        ),
        tokenize_batch_size=8,
        max_buffer_samples=64,
        decode_workers=1,
        decode_prefetch_batches=1,
        decode_fn=lambda sample: torch.full(
            (3, sample["image_height"], sample["image_width"]),
            float(sample["id"]),
        ),
        dynamic_scheduler=True,
        bucket_controller=controller,
        lookahead_per_slot=16,
        soft_buffer_limit=32,
        long_term_window_per_rank=200,
        long_term_safety_margin=0.02,
    )
    targets = []
    resolutions = []
    for _, batch in zip(range(120), batcher):
        targets.append(int(batch["bucket_scheduler"]["selected_target"]))
        resolutions.append(batch["resolution_bucket"])
    gathered_targets = [None] * world
    dist.all_gather_object(gathered_targets, targets, group=group)
    assert all(sequence == targets for sequence in gathered_targets)
    assert 10 in targets
    assert resolutions == (["square"] * 120 if rank == 0 else ["landscape"] * 120)
    assert batcher.scheduler.schedule_version >= 1
    assert batcher.scheduler.base_weights == (0, 0, 0, 0, 100)

    if rank == 0:
        print(
            "DYNAMIC BUCKET DISTRIBUTED PASS: common targets identical, "
            "rank-local resolutions differ, Gloo weight update installed at boundary",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
