import io

import torch
from PIL import Image

from uv3.eval.checkpoint_evaluator import load_reference, sample_batch


class _VAE:
    latent_channels = 4
    dtype = torch.float32

    def scale_factor(self, _image_size):
        return 8

    def decode_latents(self, latents):
        return latents[:, :3]


class _Model:
    def __init__(self):
        self.shapes = []

    def predict_velocity(self, latents, _text, _time, text_attn_mask=None):
        self.shapes.append(tuple(latents.shape))
        assert text_attn_mask is None
        return torch.zeros_like(latents)


def test_sample_batch_accepts_rectangular_resolution():
    model = _Model()
    generated = sample_batch(
        model,
        _VAE(),
        torch.zeros(1, 4, 8),
        None,
        seeds=[123],
        steps=2,
        image_height=208,
        image_width=320,
    )
    assert model.shapes == [(1, 4, 26, 40), (1, 4, 26, 40)]
    assert generated.shape == (1, 3, 26, 40)


def test_reference_uses_target_aspect_crop(tmp_path):
    encoded = io.BytesIO()
    Image.new("RGB", (640, 480), "red").save(encoded, format="JPEG")
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(encoded.getvalue())
    reference = load_reference(
        {
            "image_tar": str(image_path),
            "offset": 0,
            "size": len(encoded.getvalue()),
        },
        image_height=208,
        image_width=320,
    )
    assert reference.size == (320, 208)
