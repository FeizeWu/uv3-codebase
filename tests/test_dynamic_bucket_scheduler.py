from __future__ import annotations

import random

import pytest
import torch

from uv3.data.bucket_sampler import AspectBucket
from uv3.data.dynamic_bucket_scheduler import (
    DynamicJointBucketScheduler,
    prefix_minima_to_native_counts,
    ready_mask_from_counts,
    safe_integer_weights,
    select_shortest_common_target,
    smooth_weighted_schedule,
)
from uv3.train.fsdp2_trainer import DataPrefetcher, OnlineJointBucketBatcher


BUCKETS = (512, 640, 768, 896, 1024)


def test_ready_mask_is_always_a_monotone_suffix_for_thousands_of_inputs():
    rng = random.Random(19)
    for _ in range(5_000):
        counts = [[rng.randrange(20) for _ in BUCKETS] for _ in range(5)]
        mask = ready_mask_from_counts(counts, batch_size=12)
        bits = [bool(mask & (1 << index)) for index in range(len(BUCKETS))]
        assert bits == sorted(bits)


def test_common_ready_intersection_contains_max_target_under_pigeonhole_bound():
    rng = random.Random(23)
    max_bit = 1 << (len(BUCKETS) - 1)
    for _ in range(2_000):
        masks = []
        for _rank in range(32):
            counts = [[0] * len(BUCKETS) for _ in range(5)]
            # At least one resolution owns a complete local batch; native text
            # distribution within that resolution is adversarial.
            resolution = rng.randrange(5)
            for _sample in range(12 + rng.randrange(100)):
                counts[resolution][rng.randrange(len(BUCKETS))] += 1
            masks.append(ready_mask_from_counts(counts, batch_size=12))
        common = masks[0]
        for mask in masks[1:]:
            common &= mask
        assert common & max_bit


@pytest.mark.parametrize(
    "desired,mask,expected",
    [
        (512, 0b11111, 512),
        (512, 0b11100, 768),
        (640, 0b11000, 896),
        (896, 0b10000, 1024),
        (1024, 0b10000, 1024),
    ],
)
def test_selected_target_is_shortest_common_target_not_below_desired(
    desired, mask, expected,
):
    assert select_shortest_common_target(BUCKETS, desired, mask) == expected


def test_safe_weight_rounding_never_crosses_worst_rank_supply_minus_margin():
    rng = random.Random(29)
    window = 50_000
    margin = 0.02
    for _ in range(5_000):
        rank_prefixes = []
        for _rank in range(32):
            cuts = sorted(rng.sample(range(1, window), len(BUCKETS) - 1))
            rank_prefixes.append(cuts)
        worst_prefixes = tuple(
            min(prefix[index] for prefix in rank_prefixes)
            for index in range(len(BUCKETS) - 1)
        )
        synthetic_counts = prefix_minima_to_native_counts(worst_prefixes, window)
        weights = safe_integer_weights(
            synthetic_counts, safety_margin=margin, total_weight=100,
        )
        assert sum(weights) == 100
        assert all(weight >= 0 for weight in weights)
        for index, worst_supply in enumerate(worst_prefixes):
            demand = sum(weights[: index + 1]) / 100
            safe_supply = max(0.0, worst_supply / window - margin)
            assert demand <= safe_supply + 1e-12


def test_all_long_captions_produce_a_max_only_schedule():
    weights = safe_integer_weights((0, 0, 0, 0, 50_000), safety_margin=0.02)
    assert weights == (0, 0, 0, 0, 100)
    assert set(smooth_weighted_schedule(BUCKETS, weights)) == {1024}


def test_exact_count_windows_and_scheduler_state_round_trip():
    scheduler = DynamicJointBucketScheduler(
        BUCKETS,
        (20, 18, 18, 14, 30),
        batch_size=12,
        resolution_count=5,
        long_term_window_per_rank=17,
        safety_margin=0.02,
    )
    observations = [BUCKETS[index % len(BUCKETS)] for index in range(39)]
    for target in observations:
        scheduler.observe_native_target(target)
    first = scheduler.peek_completed_window()
    assert sum(first) == 17
    assert len(scheduler.state_dict()["completed_windows"]) == 2
    selection = scheduler.select(common_ready_mask=0b11111)
    scheduler.record_selection(selection)

    restored = DynamicJointBucketScheduler(
        BUCKETS,
        (20, 18, 18, 14, 30),
        batch_size=12,
        resolution_count=5,
        long_term_window_per_rank=17,
        safety_margin=0.02,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()
    assert restored.desired_target == scheduler.desired_target


def test_new_weights_install_only_at_period_boundary_and_advance_version():
    scheduler = DynamicJointBucketScheduler(
        BUCKETS,
        (20, 18, 18, 14, 30),
        batch_size=12,
        resolution_count=5,
        long_term_window_per_rank=10,
    )
    with pytest.raises(RuntimeError, match="schedule boundary"):
        scheduler.record_selection(scheduler.select(0b11111))
        scheduler.install_weights((0, 0, 0, 0, 100), schedule_version=1)

    scheduler = DynamicJointBucketScheduler(
        BUCKETS,
        (20, 18, 18, 14, 30),
        batch_size=12,
        resolution_count=5,
        long_term_window_per_rank=10,
    )
    for _ in range(len(scheduler.schedule)):
        scheduler.record_selection(scheduler.select(0b11111))
    assert scheduler.at_schedule_boundary
    scheduler.install_weights((0, 0, 0, 0, 100), schedule_version=1)
    assert scheduler.schedule_version == 1
    assert scheduler.desired_target == 1024


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [list(range(int(text.split(":", 1)[0]))) for text in texts]
        }


def _descriptors(count: int, *, all_long: bool = False):
    rows = []
    for index in range(count):
        length = 8 if all_long or index % 3 else 3
        resolution = "square" if index % 2 else "landscape"
        rows.append({
            "id": index,
            "text": f"{length}:sample-{index}",
            "resolution_bucket": resolution,
            "image_height": 32,
            "image_width": 32 if resolution == "square" else 48,
            "worker_id": index % 4,
            "epoch": 0,
            "shard_pos": index // 32,
            "row_group": (index // 8) % 4,
            "row_pos": index % 8,
        })
    return rows


def _batcher(samples, *, decode_prefetch_batches=1, window=64):
    return OnlineJointBucketBatcher(
        samples,
        _Tokenizer(),
        batch_size=4,
        text_buckets=(4, 8),
        text_weights=(1, 1),
        resolution_buckets=(
            AspectBucket("square", 32, 32),
            AspectBucket("landscape", 48, 32),
        ),
        tokenize_batch_size=8,
        max_buffer_samples=64,
        decode_workers=1,
        decode_prefetch_batches=decode_prefetch_batches,
        decode_fn=lambda sample: torch.full(
            (3, sample["image_height"], sample["image_width"]),
            float(sample["id"]),
        ),
        dynamic_scheduler=True,
        lookahead_per_slot=16,
        soft_buffer_limit=32,
        long_term_window_per_rank=window,
        long_term_safety_margin=0.02,
    )


def _batch_identity(batch):
    return (
        tuple(batch["text"]),
        int(batch["bucket_scheduler"]["desired_target"]),
        int(batch["bucket_scheduler"]["selected_target"]),
        bool(batch["bucket_scheduler"]["fallback"]),
        int(batch["bucket_scheduler"]["schedule_version"]),
    )


def test_all_long_distribution_falls_back_without_filling_hard_buffer():
    batcher = _batcher(_descriptors(4_000, all_long=True), window=64)
    iterator = iter(batcher)
    first = next(iterator)
    assert first["bucket_scheduler"]["desired_target"] == 4
    assert first["bucket_scheduler"]["selected_target"] == 8
    assert first["bucket_scheduler"]["fallback"] is True
    peak = first["bucket_buffer_samples"]
    emitted_ids = {
        int(text.rsplit("-", 1)[1]) for text in first["text"]
    }
    for _ in range(499):
        batch = next(iterator)
        peak = max(peak, int(batch["bucket_buffer_samples"]))
        emitted_ids.update(int(text.rsplit("-", 1)[1]) for text in batch["text"])
    iterator.close()

    assert peak <= batcher.max_buffer_samples
    assert len(emitted_ids) == 2_000
    assert batcher.scheduler.schedule_version > 0
    # With one active decode slot, selected descriptors are either emitted or
    # remain in the 2x2 rank-local queues; no descriptor disappears.
    assert batcher.source_samples == batcher.emitted_samples + batcher.buffered_samples


def test_dynamic_batcher_resume_preserves_targets_fallbacks_and_samples():
    samples = _descriptors(400)
    uninterrupted_batcher = _batcher(samples, decode_prefetch_batches=2, window=24)
    uninterrupted = iter(uninterrupted_batcher)
    expected = [_batch_identity(next(uninterrupted)) for _ in range(20)]
    uninterrupted.close()

    original = _batcher(samples, decode_prefetch_batches=2, window=24)
    iterator = iter(original)
    prefix = [_batch_identity(next(iterator)) for _ in range(7)]
    state = original.state_dict()
    source_offset = original.source_samples
    iterator.close()

    resumed = _batcher(
        samples[source_offset:], decode_prefetch_batches=2, window=24,
    )
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    suffix = [_batch_identity(next(resumed_iterator)) for _ in range(13)]
    resumed_iterator.close()

    assert prefix + suffix == expected
    assert resumed.scheduler.state_dict() == uninterrupted_batcher.scheduler.state_dict()
    assert resumed.state_dict() == uninterrupted_batcher.state_dict()


def test_descriptor_accounting_includes_pending_without_double_counting():
    batcher = _batcher(_descriptors(200), decode_prefetch_batches=2, window=24)
    iterator = iter(batcher)
    next(iterator)
    state = batcher.state_dict()
    pending_unemitted = sum(
        len(spec[2])
        for spec in state["pending_specs"]
        if not spec[4].get("emission_accounted", False)
    )
    buffered = sum(len(queue) for queue in state["queues"].values())

    assert state["source_samples"] == state["scheduled_samples"] + buffered
    assert state["scheduled_samples"] == state["emitted_samples"] + pending_unemitted
    assert all(spec[4]["control_wait_ms"] == 0.0 for spec in state["pending_specs"])
    iterator.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefetch stream required")
def test_data_prefetcher_forwards_scheduler_metadata():
    metadata = {
        "desired_target": 4,
        "selected_target": 8,
        "fallback": True,
    }
    batch = {
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "text": ["sample"],
        "bucket_scheduler": metadata,
    }
    prefetcher = DataPrefetcher(
        [batch], torch.device("cuda", 0), torch.bfloat16,
    )
    assert prefetcher.next()["bucket_scheduler"] == metadata
