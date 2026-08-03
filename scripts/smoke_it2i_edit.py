"""it2i editing semantics validation (review #1.4/#6).

Tiny MMDiT overfits edit pairs: source image -> target image is a deterministic transform
(here: horizontal flip). After overfit, sampling WITH the source as ref should reconstruct
the target (flipped source); sampling WITHOUT ref (t2i) should NOT match. This is the
decisive test that it2i ref conditioning actually drives the output (not just shape/grad).

CUDA_VISIBLE_DEVICES=6 python -m scripts.smoke_it2i_edit
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
    out = "/mnt/oss/users/wfz/uv3-codebase-runs/it2i_edit"
    os.makedirs(out, exist_ok=True)

    vae = Flux2VAE.from_pretrained(FLUX, dtype=dtype, force_upcast=False).to(dev).eval()
    mc = ModelConfig(
        architecture="mmdit", hidden_size=768, num_layers=2, num_double_layers=2,
        num_single_layers=4, num_heads=12, latent_channels=32, patch_size=2, in_channels=128,
        out_channels=128, rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16), guidance_embeds=False,
        alpha_on=False, self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(), transformer=ComponentConfig(backend="random", trainable=True),
    )
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=type("E", (), {"hidden_size": 4096})()).to(dev, dtype=dtype)
    tc = TrainConfig(optimizer=OptimizerConfig())
    opt, (nm, na) = build_optimizers(mmdit, tc)
    print(f"[it2i] MMDiT params={sum(p.numel() for p in mmdit.parameters() if p.requires_grad):,}", flush=True)

    # 4 edit pairs: source = real image, target = flipped source (deterministic edit)
    from uv3.data.parquet_dataset import ParquetImageDataset
    ds = ParquetImageDataset(root="/mnt/oss/users/lzj/imagenet-ablation/imagenet-1k",
                             parquet_glob="data/train-*.parquet", image_size=256, overfit_n=4)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    src_imgs = batch["pixel_values"].to(dev, dtype=dtype)        # (4,3,256,256) source
    tgt_imgs = srcimgs_flip = src_imgs.flip(-1)                  # target = flipped source
    with torch.no_grad():
        src_lat = vae.encode_images(src_imgs)                    # (4,32,32,32) ref latents
        tgt_lat = vae.encode_images(tgt_imgs)                    # clean target latents
        text = torch.randn(4, 64, 4096, device=dev, dtype=dtype) # dummy text (per-image fixed)
    print(f"[it2i] cached: src_lat={tuple(src_lat.shape)} tgt_lat={tuple(tgt_lat.shape)}", flush=True)

    # overfit: predict velocity on tgt_lat conditioned by src_lat (ref) + text
    from uv3.modeling.flow import interpolate, velocity_target, logit_normal_timesteps
    bs = 4
    n_ref = (src_lat.shape[-1] // 2) * (src_lat.shape[-2] // 2)   # 256 ref tokens
    t0 = time.time()
    for o in opt.values():
        o.zero_grad()
    for step in range(2000):
        t = logit_normal_timesteps(bs, dev)
        noise = torch.randn_like(tgt_lat)
        noisy = interpolate(tgt_lat, noise, t)
        pred_v = mmdit.predict_velocity(noisy, text, t, num_ref_tokens=n_ref, ref=src_lat)
        loss = torch.nn.functional.mse_loss(pred_v.float(), velocity_target(tgt_lat, noise).float())
        loss.backward()
        for o in opt.values():
            o.step()
        for o in opt.values():
            o.zero_grad()
        if step % 200 == 0:
            print(f"[it2i] step {step:4d} loss={loss.item():.4f} {(step+1)/(time.time()-t0):.1f}it/s", flush=True)

    # decisive test: sample WITH ref (it2i) vs WITHOUT ref (t2i)
    print("[it2i] sampling: it2i (with ref=source) vs t2i (no ref)...", flush=True)
    with torch.no_grad():
        # it2i: use source as ref, sample target
        sl_it2i = mmdit.sample_latents((4, 32, 32, 32), text, steps=30, device=dev, dtype=dtype)
        # redo sample loop WITH ref (sample_latents doesn't take ref; manual euler)
        from uv3.modeling.flow import euler_schedule, euler_step
        lat = torch.randn(4, 32, 32, 32, device=dev, dtype=dtype)
        times = euler_schedule(30, dev, dtype)
        for cur, foll in zip(times[:-1], times[1:]):
            v = mmdit.predict_velocity(lat, text, cur.expand(4), num_ref_tokens=n_ref, ref=src_lat)
            lat = euler_step(lat, v, cur, foll)
        it2i_sample = vae.decode_latents(lat)
        # t2i: no ref
        lat2 = torch.randn(4, 32, 32, 32, device=dev, dtype=dtype)
        for cur, foll in zip(times[:-1], times[1:]):
            v = mmdit.predict_velocity(lat2, text, cur.expand(4))
            lat2 = euler_step(lat2, v, cur, foll)
        t2i_sample = vae.decode_latents(lat2)
        # save
        from PIL import Image
        def save(img, p):
            x = img[0].clamp(-1,1).add(1).div(2).mul(255).to(torch.uint8).permute(1,2,0).cpu().numpy()
            Image.fromarray(x).save(p)
        save(src_imgs, f"{out}/source.png")
        save(tgt_imgs, f"{out}/target_flipped.png")
        save(it2i_sample, f"{out}/it2i_sample.png")
        save(t2i_sample, f"{out}/t2i_sample.png")
        # metric: it2i sample should be closer to target (flipped) than t2i
        it2i_err = (it2i_sample - tgt_imgs).abs().mean().item()
        t2i_err = (t2i_sample - tgt_imgs).abs().mean().item()
        print(f"[it2i] DONE. it2i_err={it2i_err:.4f} t2i_err={t2i_err:.4f} "
              f"(it2i should be < t2i if ref conditioning works). samples in {out}", flush=True)


if __name__ == "__main__":
    main()
