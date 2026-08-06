"""Descriptor-only adversarial soak for the production bucket state machine.

This deliberately uses a trivial tokenizer/decoder so more than the historical
8192-descriptor failure horizon can be exercised without loading model weights.
"""
from __future__ import annotations

import json

import torch

from uv3.data.bucket_sampler import AspectBucket
from uv3.data.tar_dataset import TarDecodeFailure
from uv3.train.fsdp2_trainer import OnlineJointBucketBatcher


BUCKETS = (2, 4, 6, 8, 10)
RESOLUTIONS = ("square", "landscape", "portrait", "widescreen", "phone")
OUTPUT_SAMPLES = 10_000
SOURCE_SAMPLES = 16_000


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [list(range(int(text.split(":", 1)[0]))) for text in texts]}


def _length(scenario: str, index: int, resolution_index: int) -> int:
    if scenario == "all_1024":
        return 10
    if scenario == "all_512":
        return 2
    if scenario == "normal_then_1024":
        return BUCKETS[index % 5] if index < 5_000 else 10
    if scenario == "text_resolution_correlated":
        return BUCKETS[resolution_index]
    return BUCKETS[(index * 7 + 3) % 5]


def _samples(scenario: str):
    for index in range(SOURCE_SAMPLES):
        resolution_index = (
            4 if scenario == "rare_resolution" and index % 997 == 0 else index % 4
        )
        yield {
            "id": index,
            "text": f"{_length(scenario, index, resolution_index)}:{scenario}-{index}",
            "resolution_bucket": RESOLUTIONS[resolution_index],
            "image_height": 8,
            "image_width": 8,
            "worker_id": index % 4,
            "epoch": 0,
            "shard_pos": index // 1_000,
            "row_group": (index // 100) % 10,
            "row_pos": index % 100,
        }


def _run(scenario: str) -> dict:
    def decode(sample):
        if scenario == "malformed_replacement" and sample["id"] % 997 == 0:
            return TarDecodeFailure("synthetic malformed image")
        return torch.full((3, 8, 8), float(sample["id"]))

    batcher = OnlineJointBucketBatcher(
        _samples(scenario),
        _Tokenizer(),
        batch_size=4,
        text_buckets=BUCKETS,
        text_weights=(20, 18, 18, 14, 30),
        resolution_buckets=tuple(
            AspectBucket(name, 8, 8) for name in RESOLUTIONS
        ),
        tokenize_batch_size=64,
        max_buffer_samples=256,
        decode_workers=2,
        decode_prefetch_batches=2,
        decode_fn=decode,
        dynamic_scheduler=True,
        lookahead_per_slot=64,
        soft_buffer_limit=128,
        long_term_window_per_rank=500,
        long_term_safety_margin=0.02,
    )
    iterator = iter(batcher)
    batches = OUTPUT_SAMPLES // batcher.batch_size
    max_buffer = 0
    for _ in range(batches):
        batch = next(iterator)
        max_buffer = max(max_buffer, int(batch["bucket_buffer_samples"]))
    state = batcher.state_dict()
    pending_unemitted = sum(
        len(spec[2])
        for spec in state["pending_specs"]
        if not spec[4].get("emission_accounted", False)
    )
    buffered = sum(len(queue) for queue in state["queues"].values())
    assert state["source_samples"] == (
        state["scheduled_samples"] + buffered + state["decode_error_samples"]
    )
    assert state["scheduled_samples"] == state["emitted_samples"] + pending_unemitted
    assert state["emitted_samples"] == OUTPUT_SAMPLES
    assert max_buffer <= 256
    assert batcher.scheduler.schedule_version > 0
    iterator.close()
    return {
        "scenario": scenario,
        "source": state["source_samples"],
        "emitted": state["emitted_samples"],
        "buffer_peak": max_buffer,
        "decode_errors": state["decode_error_samples"],
        "fallback_steps": batcher.scheduler.fallback_steps,
        "schedule_version": batcher.scheduler.schedule_version,
        "weights": list(batcher.scheduler.base_weights),
    }


def main() -> None:
    scenarios = (
        "all_1024",
        "all_512",
        "normal_then_1024",
        "text_resolution_correlated",
        "rare_resolution",
        "malformed_replacement",
    )
    results = [_run(scenario) for scenario in scenarios]
    print(json.dumps({"status": "PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()
