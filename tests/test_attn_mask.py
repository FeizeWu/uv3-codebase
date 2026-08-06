"""Attention mask verification (task 1.3).

①全 1 mask == 无 mask,逐元素一致;②同 batch 带 pad vs 手工截断去 pad 两路输出一致。
Run: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tests/test_attn_mask.py
"""
from __future__ import annotations

import torch

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig
from uv3.modeling.mmdit import MMDiT
from uv3.train.fsdp2_trainer import _align_text_to_joint_length, _attention_mask


def _make_model(dev, dtype=torch.float32):
    mc = ModelConfig(
        architecture="mmdit", hidden_size=256, num_layers=1, num_double_layers=1,
        num_single_layers=1, num_heads=4, latent_channels=32, patch_size=2, in_channels=128,
        out_channels=128, rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16),
        guidance_embeds=False, flex_attention=False, alpha_on=False,
        self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(),
        transformer=ComponentConfig(backend="random", trainable=True),
    )
    stub = type("E", (), {"hidden_size": 256})
    return MMDiT.build(mc.transformer, mc, text_encoder=stub()).to(dev, dtype=dtype)


def test_all_ones_mask_equals_no_mask():
    """全 1 mask == 无 mask,逐元素一致。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev, dtype=torch.float32)
    m.eval()
    b, c, h, w = 2, 32, 16, 16
    n_txt = 8
    noisy = torch.randn(b, c, h, w, device=dev)
    text = torch.randn(b, n_txt, 256, device=dev)
    t = torch.tensor([0.3, 0.7], device=dev)

    # no mask
    with torch.no_grad():
        pred_none = m.predict_velocity(noisy, text, t)

    # all-ones mask (no padding)
    n_img = (h // 2) * (w // 2)
    mask_ones = torch.zeros(b, 1, 1, n_txt + n_img, device=dev)  # all 0 = all attend
    with torch.no_grad():
        pred_masked = m.predict_velocity(noisy, text, t, text_attn_mask=mask_ones)

    diff = (pred_none - pred_masked).abs().max().item()
    print(f"  all-ones mask vs no mask: diff={diff:.2e} (target < 1e-5)")
    assert diff < 1e-5, f"all-ones mask should equal no mask: diff={diff}"


def test_padded_batch_vs_truncated():
    """同 batch 带 pad vs 手工截断去 pad 两路输出一致。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev, dtype=torch.float32)
    m.eval()
    b, c, h, w = 2, 32, 16, 16
    n_img = (h // 2) * (w // 2)

    # sample 0 has 6 real txt tokens, sample 1 has 8 (pad sample 0 to 8)
    n_txt_padded = 8
    n_txt_real_0 = 6  # sample 0 has 2 pad tokens

    text_padded = torch.randn(b, n_txt_padded, 256, device=dev)
    noisy = torch.randn(b, c, h, w, device=dev)
    t = torch.tensor([0.5, 0.5], device=dev)

    # mask: sample 0 has pad at positions 6,7
    mask = torch.zeros(b, 1, 1, n_txt_padded + n_img, device=dev)
    mask[0, 0, 0, n_txt_real_0:n_txt_padded] = -65504.0

    with torch.no_grad():
        pred_padded = m.predict_velocity(noisy, text_padded, t, text_attn_mask=mask)

    # truncated: sample 0 with only 6 tokens (run individually then stack)
    text_trunc_0 = text_padded[0:1, :n_txt_real_0]
    text_trunc_1 = text_padded[1:1+1]  # 8 tokens
    with torch.no_grad():
        # sample 0 with 6 tokens (no mask needed)
        pred_0 = m.predict_velocity(noisy[0:1], text_trunc_0, t[0:1])
        # sample 1 with 8 tokens
        pred_1 = m.predict_velocity(noisy[1:2], text_trunc_1, t[1:2])

    # compare sample 0 (padded vs truncated) — should be very close
    diff_0 = (pred_padded[0] - pred_0[0]).abs().max().item()
    # sample 1 (no pad) should be identical
    diff_1 = (pred_padded[1] - pred_1[0]).abs().max().item()
    print(f"  sample 0 (padded vs truncated): diff={diff_0:.2e}")
    print(f"  sample 1 (no pad, identical): diff={diff_1:.2e}")
    # CUDA may select a different GEMM kernel for batch=2 and batch=1, so even
    # the unpadded control is not bit-identical. Both paths should nevertheless
    # agree within normal float32/TF32 kernel-ordering error.
    assert diff_1 < 2e-3, f"sample 1 (no pad) should be numerically equal: diff={diff_1}"
    # Tighten the old 0.1 tolerance: masked padding and a truly shorter text
    # sequence must be equivalent at the same numerical tolerance.
    assert diff_0 < 2e-3, f"padded sample 0 should match truncated: diff={diff_0}"


def test_joint_length_alignment_padding_is_numerically_inert():
    """Resolution-alignment slots are masked and cannot change image output."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev, dtype=torch.float32)
    m.eval()
    noisy = torch.randn(1, 32, 32, 32, device=dev)
    text = torch.randn(1, 8, 256, device=dev)
    text_valid = torch.ones(1, 8, dtype=torch.long, device=dev)
    t = torch.tensor([0.5], device=dev)
    n_img = (noisy.shape[-2] // 2) * (noisy.shape[-1] // 2)

    aligned_text, aligned_valid, alignment = _align_text_to_joint_length(
        text, text_valid, image_tokens=n_img, image_token_budget=n_img + 4,
    )
    aligned_mask = _attention_mask(m, aligned_valid, n_img, 128, dev)

    with torch.no_grad():
        baseline = m.predict_velocity(noisy, text, t)
        aligned = m.predict_velocity(
            noisy, aligned_text, t, text_attn_mask=aligned_mask,
        )

    diff = (baseline - aligned).abs().max().item()
    assert alignment == 4
    assert diff < 1e-5, f"masked alignment slots changed output: diff={diff}"


def test_rank_local_rectangles_share_joint_length_and_preserve_output_shape():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev, dtype=torch.float32)
    m.eval()
    joint_lengths = []
    for latent_height, latent_width in ((32, 32), (26, 40), (24, 42)):
        noisy = torch.randn(1, 32, latent_height, latent_width, device=dev)
        text = torch.randn(1, 8, 256, device=dev)
        valid = torch.ones(1, 8, dtype=torch.long, device=dev)
        image_tokens = (latent_height // 2) * (latent_width // 2)
        text, valid, _ = _align_text_to_joint_length(
            text, valid, image_tokens=image_tokens, image_token_budget=260,
        )
        mask = _attention_mask(m, valid, image_tokens, 128, dev)
        with torch.no_grad():
            output = m.predict_velocity(
                noisy, text, torch.tensor([0.5], device=dev), text_attn_mask=mask,
            )
        assert output.shape == noisy.shape
        joint_lengths.append(text.shape[1] + image_tokens)
    assert joint_lengths == [268, 268, 268]


if __name__ == "__main__":
    print("test_attn_mask: mask verification")
    test_all_ones_mask_equals_no_mask()
    print("  ✓ all-ones mask == no mask")
    test_padded_batch_vs_truncated()
    print("  ✓ padded batch vs truncated")
    print("test_attn_mask: ALL PASS")
