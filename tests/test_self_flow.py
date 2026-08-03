"""Golden equivalence test for forward_per_token (round 5 step 5.1).

CRITICAL: if token_timesteps all == same scalar t, forward_per_token output must match
the original transformer.forward() path element-wise (atol 1e-5 fp32). This proves the
orchestration is correct. If this fails, the implementation has a bug — do not proceed.

Run: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tests/test_self_flow.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig
from uv3.modeling.mmdit import MMDiT
from uv3.modeling.flow import interpolate, velocity_target, logit_normal_timesteps
from uv3.modeling.self_flow import build_self_flow_latents_continuous


def _make_model(dev, dtype=torch.float32):
    """Tiny model in fp32 for precise equivalence check."""
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
    m = MMDiT.build(mc.transformer, mc, text_encoder=stub()).to(dev, dtype=dtype)
    return m


def test_equivalence_scalar_vs_per_token():
    """token_timesteps all == t → forward_per_token == transformer.forward (atol 1e-5)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev, dtype=torch.float32)
    m.eval()
    b, c, h, w = 2, 32, 16, 16  # small for speed
    n_txt = 8

    noisy = torch.randn(b, c, h, w, device=dev)
    text = torch.randn(b, n_txt, 256, device=dev)
    t = torch.tensor([0.3, 0.7], device=dev)

    with torch.no_grad():
        # original path
        pred_scalar = m.predict_velocity(noisy, text, t)

        # per-token path with all token_timesteps == t (should be identical)
        from uv3.modeling.vae import patchify_latents
        packed = patchify_latents(noisy)
        n_img = packed.shape[-2] * packed.shape[-1]
        token_t = t.unsqueeze(1).expand(b, n_img).contiguous()
        pred_per_token = m.predict_velocity(noisy, text, t, token_timesteps=token_t)

    diff = (pred_scalar - pred_per_token).abs().max().item()
    print(f"  equivalence diff = {diff:.2e} (target < 1e-5)")
    assert diff < 1e-5, f"forward_per_token != transformer.forward: diff={diff}"


def test_token_timesteps_none_zero_regression():
    """token_timesteps=None → original path (smoke, no crash)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev)
    noisy = torch.randn(1, 32, 16, 16, device=dev)
    text = torch.randn(1, 8, 256, device=dev)
    t = torch.tensor([0.5], device=dev)
    with torch.no_grad():
        pred = m.predict_velocity(noisy, text, t, token_timesteps=None)
    assert pred.shape == noisy.shape


def test_mask_ratio_zero_returns_student_full():
    """mask_ratio=0 → student_latents == interpolate(clean,noise,t) (no paired contamination)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clean = torch.randn(2, 32, 16, 16, device=dev)
    noise = torch.randn_like(clean)
    t = torch.tensor([0.3, 0.7], device=dev)
    student, teacher, t_teach, token_t = build_self_flow_latents_continuous(
        clean, noise, t, mask_ratio=0.0, ratio=0.5, timestep_mode="ratio",
    )
    expected = interpolate(clean, noise, t)
    diff = (student - expected).abs().max().item()
    print(f"  mask_ratio=0 diff = {diff:.2e} (target < 1e-6)")
    assert diff < 1e-6, f"mask_ratio=0 should return student_full, diff={diff}"
    # token_timesteps should all be t when mask_ratio=0
    tt_diff = (token_t - t.unsqueeze(1).expand_as(token_t)).abs().max().item()
    assert tt_diff < 1e-6, f"token_timesteps should all be t when mask_ratio=0"


def test_it2i_ref_zero_timestep():
    """it2i with ref: ref segment token_timesteps=0, forward doesn't crash, shape correct."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _make_model(dev)
    noisy = torch.randn(1, 32, 16, 16, device=dev)
    ref = torch.randn(1, 32, 16, 16, device=dev)
    text = torch.randn(1, 8, 256, device=dev)
    t = torch.tensor([0.5], device=dev)
    n_ref = (16 // 2) * (16 // 2)  # 64
    token_t = torch.full((1, n_ref), 0.5, device=dev)  # img tokens at t=0.5
    with torch.no_grad():
        pred = m.predict_velocity(noisy, text, t, num_ref_tokens=n_ref, ref=ref, token_timesteps=token_t)
    assert pred.shape == noisy.shape, f"shape {pred.shape} != {noisy.shape}"


if __name__ == "__main__":
    print("test_self_flow: golden equivalence + API checks")
    test_equivalence_scalar_vs_per_token()
    print("  ✓ equivalence (forward_per_token == transformer.forward)")
    test_token_timesteps_none_zero_regression()
    print("  ✓ token_timesteps=None zero regression")
    test_mask_ratio_zero_returns_student_full()
    print("  ✓ mask_ratio=0 returns student_full")
    test_it2i_ref_zero_timestep()
    print("  ✓ it2i ref with per-token timestep")
    print("test_self_flow: ALL PASS")
