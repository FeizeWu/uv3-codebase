import itertools

import torch
from PIL import Image

from uv3.data.bucket_sampler import (
    AspectBucket,
    build_aspect_buckets,
    choose_aspect_bucket,
    normalize_bucket_names,
)
from uv3.data.transforms import center_crop_resize, pil_to_tensor
from uv3.train.fsdp2_trainer import (
    LengthBucketBatcher,
    OnlineJointBucketBatcher,
    _align_text_to_joint_length,
    _smooth_weighted_schedule,
    format_resolution_bucket_loss,
)


def test_mar_alias_builds_equal_area_stride_aligned_buckets():
    buckets = build_aspect_buckets(256, "mar_256", stride=16)
    assert [bucket.name for bucket in buckets] == [
        "square", "landscape", "portrait", "widescreen", "phone",
    ]
    assert [(bucket.width, bucket.height) for bucket in buckets] == [
        (256, 256), (320, 208), (208, 320), (336, 192), (192, 336),
    ]
    assert all(bucket.width % 16 == bucket.height % 16 == 0 for bucket in buckets)
    assert all(abs(bucket.width * bucket.height / 256**2 - 1.0) < 0.03 for bucket in buckets)


def test_bucket_selection_uses_log_aspect_distance():
    buckets = build_aspect_buckets(256)
    assert choose_aspect_bucket(1600, 900, buckets).name == "widescreen"
    assert choose_aspect_bucket(900, 1600, buckets).name == "phone"
    assert choose_aspect_bucket(1200, 800, buckets).name == "landscape"
    assert choose_aspect_bucket(800, 1200, buckets).name == "portrait"
    assert normalize_bucket_names("square,portrait") == ("square", "portrait")


def test_rectangular_center_crop_and_tensor_shape():
    image = Image.new("RGB", (1200, 800), (10, 20, 30))
    cropped = center_crop_resize(image, (208, 320))
    tensor = pil_to_tensor(image, (208, 320))
    assert cropped.size == (320, 208)
    assert tensor.shape == (3, 208, 320)
    assert tensor.dtype == torch.float32


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(int(text)))}


def _joint_samples():
    shapes = {"square": (32, 32), "landscape": (16, 48)}
    rows = []
    for _ in range(16):
        for resolution, text_length in itertools.product(shapes, (4, 12)):
            height, width = shapes[resolution]
            rows.append({
                "pixel_values": torch.zeros(3, height, width),
                "text": str(text_length),
                "resolution_bucket": resolution,
                "image_height": height,
                "image_width": width,
            })
    return rows


def _make_batcher():
    return LengthBucketBatcher(
        _joint_samples(),
        _Tokenizer(),
        batch_size=2,
        buckets=(8, 16),
        weights=(1, 1),
        resolution_buckets=(
            AspectBucket("square", 32, 32),
            AspectBucket("landscape", 48, 16),
        ),
        resolution_weights=(1, 1),
    )


def test_joint_bucket_schedule_is_rank_deterministic_and_batches_are_uniform():
    first = iter(_make_batcher())
    second = iter(_make_batcher())
    for _ in range(8):
        left, right = next(first), next(second)
        assert left["resolution_bucket"] == right["resolution_bucket"]
        assert left["text_bucket_length"] == right["text_bucket_length"]
        assert left["pixel_values"].shape == right["pixel_values"].shape
        assert left["pixel_values"].shape[0] == 2
        assert left["input_ids"].shape == (2, left["text_bucket_length"])


def test_smooth_schedule_has_exact_weights_and_low_prefix_discrepancy():
    schedule = _smooth_weighted_schedule((4, 8, 12), (2, 3, 5))
    assert {bucket: schedule.count(bucket) for bucket in (4, 8, 12)} == {
        4: 2, 8: 3, 12: 5,
    }
    for prefix_length in range(1, len(schedule) + 1):
        prefix = schedule[:prefix_length]
        for bucket, weight in zip((4, 8, 12), (2, 3, 5)):
            expected = prefix_length * weight / 10
            assert abs(prefix.count(bucket) - expected) <= 1


class _BatchTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [list(range(int(text))) for text in texts]}


def _descriptor_samples(resolution, height, width):
    return [
        {
            "text": "3",
            "resolution_bucket": resolution,
            "image_height": height,
            "image_width": width,
        }
        for _ in range(12)
    ]


def _decode_descriptor(sample):
    return torch.zeros(3, sample["image_height"], sample["image_width"])


def _online_batcher(samples):
    return OnlineJointBucketBatcher(
        samples,
        _BatchTokenizer(),
        batch_size=2,
        text_buckets=(4, 8),
        text_weights=(1, 1),
        resolution_buckets=(
            AspectBucket("square", 32, 32),
            AspectBucket("landscape", 48, 16),
        ),
        tokenize_batch_size=4,
        max_buffer_samples=100,
        decode_workers=1,
        decode_prefetch_batches=1,
        decode_fn=_decode_descriptor,
    )


def test_online_joint_schedule_allows_rank_local_aspect_and_caption_promotion():
    square = iter(_online_batcher(_descriptor_samples("square", 32, 32)))
    landscape = iter(_online_batcher(_descriptor_samples("landscape", 16, 48)))
    first_square, first_landscape = next(square), next(landscape)
    second_square, second_landscape = next(square), next(landscape)

    assert first_square["text_bucket_length"] == first_landscape["text_bucket_length"] == 4
    assert first_square["joint_token_length"] == first_landscape["joint_token_length"] == 8
    assert first_square["resolution_bucket"] == "square"
    assert first_landscape["resolution_bucket"] == "landscape"
    assert first_square["pixel_values"].shape[-2:] == (32, 32)
    assert first_landscape["pixel_values"].shape[-2:] == (16, 48)
    assert second_square["text_bucket_length"] == second_landscape["text_bucket_length"] == 8
    assert second_square["bucket_promoted_samples"] == 2
    assert second_landscape["bucket_promoted_samples"] == 2


def test_joint_alignment_adds_only_invalid_slots():
    text = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.long)
    aligned_text, aligned_mask, padding = _align_text_to_joint_length(
        text, mask, image_tokens=252, image_token_budget=260,
    )
    assert padding == 8
    assert aligned_text.shape == (2, 16, 16)
    assert torch.equal(aligned_text[:, :8], text)
    assert aligned_mask[:, :8].all()
    assert not aligned_mask[:, 8:].any()


def test_resolution_bucket_loss_formats_global_sums():
    buckets = (
        AspectBucket("square", 32, 32),
        AspectBucket("landscape", 48, 16),
    )
    reduced = torch.tensor([
        [4.0, 0.0],
        [8.0, 0.0],
        [2.0, 0.0],
    ], dtype=torch.float64)
    result = format_resolution_bucket_loss(
        reduced, buckets, sf_enabled=True, sf_coeff=0.8,
    )
    assert result["names"] == ["square", "landscape"]
    assert result["width"] == [32, 48]
    assert result["height"] == [32, 16]
    assert result["count"] == [4, 0]
    assert result["fm_loss"] == [2.0, 0.0]
    assert result["self_flow_loss"] == [0.5, 0.0]
    assert result["total_loss"] == [2.4, 0.0]
