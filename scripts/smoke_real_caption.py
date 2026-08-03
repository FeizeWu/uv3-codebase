"""真 caption 决定性测试(task 3.2).

从 0803-test 抽 8 张带 caption 的图(含长 caption)→ 真 Qwen3.5 编码过拟合 2k 步
→ 正确 caption 采样 vs 换别图 caption 采样 → 肉眼可分(前者复现该图、后者偏离)。

CUDA_VISIBLE_DEVICES=6 python -m scripts.smoke_real_caption
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig, TrainConfig, OptimizerConfig
from uv3.modeling.vae import Flux2VAE
from uv3.modeling.qwen3_5 import Qwen3_5TextEncoder
from uv3.modeling.mmdit import MMDiT
from uv3.optim.build_optimizers import build_optimizers
from uv3.data.tar_dataset import TarMetadataDataset

FLUX = "/mnt/data/share/checkpoints/black-forest-labs/FLUX.2-dev"
QWEN = "/mnt/data/share/checkpoints/Qwen/Qwen3.5-9B"


def main():
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    out = "/mnt/oss/users/wfz/uv3-codebase-runs/real_caption_test"
    os.makedirs(out, exist_ok=True)

    # --- 1. Load models ---
    vae = Flux2VAE.from_pretrained(FLUX, dtype=dtype, force_upcast=False).to(dev).eval()
    qwen = Qwen3_5TextEncoder.from_pretrained(QWEN, max_length=1024, dtype=dtype).to(dev).eval()

    mc = ModelConfig(
        architecture="mmdit", hidden_size=768, num_layers=2, num_double_layers=2,
        num_single_layers=4, num_heads=12, latent_channels=32, patch_size=2, in_channels=128,
        out_channels=128, rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16),
        guidance_embeds=False, flex_attention=False, alpha_on=False,
        self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(),
        transformer=ComponentConfig(backend="random", trainable=True),
    )
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=qwen).to(dev, dtype=dtype)
    tc = TrainConfig(optimizer=OptimizerConfig())
    opt, (nm, na) = build_optimizers(mmdit, tc)
    print(f"[rc] MMDiT params={sum(p.numel() for p in mmdit.parameters() if p.requires_grad):,}", flush=True)

    # --- 2. Load 8 images with captions from 0803-test ---
    ds = TarMetadataDataset(image_size=256)
    samples = []
    for s in ds:
        samples.append(s)
        if len(samples) >= 8:
            break
    print(f"[rc] loaded {len(samples)} samples", flush=True)
    for i, s in enumerate(samples):
        print(f"  [{i}] caption len={len(s['text'])} : {s['text'][:80]}...", flush=True)

    # cache latents + text features
    imgs = torch.stack([s["pixel_values"] for s in samples]).to(dev, dtype=dtype)
    captions = [s["text"] for s in samples]
    with torch.no_grad():
        latents = vae.encode_images(imgs)
        ids, mask = qwen.tokenize(captions, dev)
        text_feats = qwen.encode_text(ids, mask)
    # build attn mask
    n_txt = text_feats.shape[1]
    n_img = (latents.shape[-1] // 2) * (latents.shape[-2] // 2)
    full_mask = torch.zeros(8, 1, 1, n_txt + n_img, device=dev, dtype=dtype)
    pad_len = n_txt - mask.sum(dim=1)
    for i in range(8):
        if pad_len[i] > 0:
            full_mask[i, 0, 0, n_txt - int(pad_len[i].item()):n_txt] = -65504.0
    print(f"[rc] cached: latents={tuple(latents.shape)} text={tuple(text_feats.shape)} n_txt={n_txt}", flush=True)

    # --- 3. Overfit 2k steps ---
    bs = 8
    from uv3.modeling.flow import interpolate, velocity_target, logit_normal_timesteps
    t0 = time.time()
    for o in opt.values():
        o.zero_grad()
    for step in range(2000):
        t = logit_normal_timesteps(bs, dev)
        noise = torch.randn_like(latents)
        noisy = interpolate(latents, noise, t)
        pred_v = mmdit.predict_velocity(noisy, text_feats, t, text_attn_mask=full_mask)
        loss = torch.nn.functional.mse_loss(pred_v.float(), velocity_target(latents, noise).float())
        loss.backward()
        for o in opt.values():
            o.step()
        for o in opt.values():
            o.zero_grad()
        if step % 200 == 0:
            print(f"[rc] step {step:4d} loss={loss.item():.4f} {(step+1)/(time.time()-t0):.1f}it/s", flush=True)

    # --- 4. Decisive test: correct caption vs swapped caption ---
    print("[rc] sampling with correct vs swapped captions...", flush=True)
    from uv3.modeling.flow import euler_schedule, euler_step
    with torch.no_grad():
        times = euler_schedule(30, dev, dtype)
        # correct caption: sample image 0 with caption 0
        lat_0 = torch.randn(1, *latents.shape[1:], device=dev, dtype=dtype)
        for cur, foll in zip(times[:-1], times[1:]):
            v = mmdit.predict_velocity(lat_0, text_feats[0:1], cur.expand(1), text_attn_mask=full_mask[0:1])
            lat_0 = euler_step(lat_0, v, cur, foll)
        img_correct = vae.decode_latents(lat_0)

        # swapped caption: sample image 0 with caption 4 (different image's caption)
        lat_1 = lat_0.clone()  # same noise start
        # actually use fresh noise for fair comparison
        lat_1 = torch.randn(1, *latents.shape[1:], device=dev, dtype=dtype)
        torch.manual_seed(42)  # reset to get same noise
        lat_0 = torch.randn(1, *latents.shape[1:], device=dev, dtype=dtype)
        lat_1 = lat_0.clone()
        for cur, foll in zip(times[:-1], times[1:]):
            v0 = mmdit.predict_velocity(lat_0, text_feats[0:1], cur.expand(1), text_attn_mask=full_mask[0:1])
            lat_0 = euler_step(lat_0, v0, cur, foll)
            v1 = mmdit.predict_velocity(lat_1, text_feats[4:5], cur.expand(1), text_attn_mask=full_mask[4:5])
            lat_1 = euler_step(lat_1, v1, cur, foll)
        img_correct = vae.decode_latents(lat_0)
        img_swapped = vae.decode_latents(lat_1)

        # save
        from PIL import Image
        def save(img, p):
            x = img[0].clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(x).save(p)
        # save reference + both samples
        ref_imgs = vae.decode_latents(latents[:1])
        save(ref_imgs, f"{out}/reference_0.png")
        save(img_correct, f"{out}/sample_correct_caption.png")
        save(img_swapped, f"{out}/sample_swapped_caption.png")

        # metric: correct should be closer to reference than swapped
        err_correct = (img_correct - ref_imgs).abs().mean().item()
        err_swapped = (img_swapped - ref_imgs).abs().mean().item()
        print(f"[rc] DONE. correct_err={err_correct:.4f} swapped_err={err_swapped:.4f} "
              f"(correct should be < swapped). samples in {out}", flush=True)


if __name__ == "__main__":
    main()
