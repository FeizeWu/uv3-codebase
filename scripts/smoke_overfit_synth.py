"""Synthetic overfit: random clean latents (no parquet, no VAE-encode-in-loop), pure MMDiT
convergence test. Only 1 VAE decode at the end (to produce a viewable image, not for training).
Isolates whether the overfit hang is in VAE-encode/parquet vs the MMDiT loop.

CUDA_VISIBLE_DEVICES=5 python -m scripts.smoke_overfit_synth
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig, TrainConfig, OptimizerConfig
from uv3.modeling.vae import Flux2VAE
from uv3.modeling.mmdit import MMDiT
from uv3.optim.build_optimizers import build_optimizers

FLUX = "/mnt/data/share/checkpoints/black-forest-labs/FLUX.2-dev"


def main():
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    out = "/mnt/oss/users/wfz/uv3-codebase-runs/overfit_synth"
    os.makedirs(out, exist_ok=True)

    vae = Flux2VAE.from_pretrained(FLUX, dtype=dtype, force_upcast=False).to(dev).eval()
    mc = ModelConfig(
        architecture="mmdit", hidden_size=768, num_layers=2, num_double_layers=2,
        num_single_layers=2, num_heads=12, latent_channels=32, patch_size=2, in_channels=128,
        out_channels=128, rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16), guidance_embeds=False,
        alpha_on=False, self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(), transformer=ComponentConfig(backend="random", trainable=True),
    )
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=type("E", (), {"hidden_size": 4096})()).to(dev, dtype=dtype)
    if os.environ.get("UV3_COMPILE"):
        mmdit.transformer = torch.compile(mmdit.transformer)
        print("[synth] torch.compile ON (mmdit.transformer)", flush=True)
    tc = TrainConfig(optimizer=OptimizerConfig())
    opt, (nm, na) = build_optimizers(mmdit, tc)
    print(f"[synth] MMDiT params={sum(p.numel() for p in mmdit.parameters() if p.requires_grad):,} muon={nm} adam={na}", flush=True)

    # synthetic clean latents (already in VAE-latent space) + dummy text (4096-dim)
    N = 16
    clean = torch.randn(N, 32, 32, 32, device=dev, dtype=dtype) * 2.0
    text = torch.randn(N, 64, 4096, device=dev, dtype=dtype)
    bs = 8
    init = None
    t0 = time.time()
    for o in opt.values():
        o.zero_grad()
    for step in range(int(os.environ.get("UV3_STEPS", "1000"))):
        idx = torch.randint(0, N, (bs,))
        loss = mmdit.training_loss(clean[idx], text[idx])
        loss.backward()
        if init is None:
            init = loss.item()
        for o in opt.values():
            o.step()
        for o in opt.values():
            o.zero_grad()
        if step % 10 == 0:
            print(f"[synth] step {step:3d} loss={loss.item():.4f} init={init:.4f} ratio={loss.item()/init:.3f} {(step+1)/(time.time()-t0):.1f}it/s", flush=True)

    # sample 4 + decode (1 VAE decode, viewable image)
    with torch.no_grad():
        sl = mmdit.sample_latents((4, 32, 32, 32), text[:4], steps=30, device=dev, dtype=dtype)
        imgs = vae.decode_latents(sl)
        from PIL import Image
        x = imgs[0].clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(x).save(os.path.join(out, "synth_sample.png"))
    print(f"[synth] DONE final ratio={loss.item()/init:.3f} (target<0.05). sample in {out}", flush=True)


if __name__ == "__main__":
    main()
