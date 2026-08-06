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
from uv3.train.fsdp2_trainer import LengthBucketBatcher


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
