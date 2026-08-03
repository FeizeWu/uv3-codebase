"""Single-GPU trainer with cached overfit mode.

Pre-encodes overfit_n images' latents + text features once, then trains only the MMDiT
(no per-step VAE/Qwen cost) -> fast overfit. Multi-GPU FSDP2 path is in fsdp2.py.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.parquet_dataset import ParquetImageDataset
from ..modeling.vae import Flux2VAE
from ..modeling.qwen3_5 import Qwen3_5TextEncoder
from ..modeling.mmdit import MMDiT
from ..optim.build_optimizers import build_optimizers


def save_png(imgs: torch.Tensor, path: str):
    """imgs: (B,3,H,W) in [-1,1] -> save first as PNG via PIL."""
    from PIL import Image
    x = imgs[0].clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)


def build_models(cfg: ExperimentConfig, dev, dtype, dummy_text=False):
    vae = Flux2VAE.from_pretrained(cfg.model.vae.pretrained, dtype=dtype, force_upcast=False).to(dev).eval()
    qwen = None
    if not dummy_text:
        qwen = Qwen3_5TextEncoder.from_pretrained(cfg.model.qwen_vl.pretrained, max_length=cfg.model.qwen_vl.max_length, dtype=dtype).to(dev).eval()
    # text dim: real Qwen=4096, or stub 4096 for dummy
    text_dim = (qwen.hidden_size if qwen else 4096)
    class _Stub:
        hidden_size = text_dim
    mmdit = MMDiT.build(cfg.model.transformer, cfg.model, text_encoder=(qwen or _Stub())).to(dev, dtype=dtype)
    optimizers, (n_m, n_a) = build_optimizers(mmdit, cfg)
    return vae, qwen, mmdit, optimizers, (n_m, n_a)


@torch.no_grad()
def cache_overfit_set(vae, qwen, cfg, dev, dtype, dummy_text=False):
    """Encode overfit_n images once -> cached latents + text feats."""
    n = cfg.data.overfit_n if hasattr(cfg.data, "overfit_n") else 256
    ds = ParquetImageDataset(
        root=cfg.data.root, split=cfg.data.split, parquet_glob=cfg.data.parquet_glob,
        image_size=cfg.data.image_size, overfit_n=n,
        image_field=cfg.data.image_field, label_field=getattr(cfg.data, "label_field", "label"),
    )
    loader = DataLoader(ds, batch_size=min(16, n), num_workers=2)
    lats, txts = [], []
    g = torch.Generator(dev).manual_seed(123)
    have = 0
    for batch in loader:
        pv = batch["pixel_values"].to(dev, dtype=dtype)
        lats.append(vae.encode_images(pv))
        if dummy_text:
            txts.append(torch.randn(pv.shape[0], 64, 4096, device=dev, dtype=dtype, generator=g))
        else:
            ids, mask = qwen.tokenize(batch["text"], dev)
            txts.append(qwen.encode_text(ids, mask))
        have += pv.shape[0]
        if have >= n:           # overfit dataset is infinite (while True) -> must break
            break
    latents = torch.cat(lats)[:n]
    text = torch.cat(txts)[:n]
    return latents, text


def run_overfit(cfg: ExperimentConfig, dev=None, dtype=torch.bfloat16, dummy_text=True):
    dev = dev or torch.device("cuda:0")
    torch.manual_seed(cfg.train.seed)
    out_dir = os.path.join(cfg.train.output_dir, cfg.train.run_name)
    os.makedirs(out_dir, exist_ok=True)

    vae, qwen, mmdit, optimizers, (n_m, n_a) = build_models(cfg, dev, dtype, dummy_text=dummy_text)
    n_params = sum(p.numel() for p in mmdit.parameters() if p.requires_grad)
    print(f"[overfit] MMDiT trainable params: {n_params:,} | muon tensors={n_m} adam={n_a} | dummy_text={dummy_text}", flush=True)

    print("[overfit] caching overfit set ...", flush=True)
    latents, text = cache_overfit_set(vae, qwen, cfg, dev, dtype, dummy_text=dummy_text)
    print(f"[overfit] cached: latents={tuple(latents.shape)} text={tuple(text.shape)}", flush=True)

    bs = cfg.train.batch_size_per_gpu
    N = latents.shape[0]
    max_steps = cfg.train.max_steps
    log_every = cfg.train.log_every
    grad_accum = max(1, cfg.train.grad_accum)
    init_loss = None
    t0 = time.time()
    for o in optimizers.values():
        o.zero_grad()

    for step in range(max_steps):
        idx = torch.randint(0, N, (bs,))
        clean = latents[idx]
        txt = text[idx]
        loss = mmdit.training_loss(clean, txt)
        (loss / grad_accum).backward()
        if init_loss is None:
            init_loss = loss.item()
        if (step + 1) % grad_accum == 0:
            for o in optimizers.values():
                o.step()
            for o in optimizers.values():
                o.zero_grad()
        if step % log_every == 0:
            ratio = (loss.item() / init_loss) if init_loss else float("nan")
            spd = (step + 1) / (time.time() - t0 + 1e-9)
            print(f"[overfit] step {step:4d} loss={loss.item():.4f} (init={init_loss:.4f}, ratio={ratio:.3f}) {spd:.2f}it/s", flush=True)
        # intermediate sample every 2000 steps to watch structure emerge
        if step > 0 and step % 2000 == 0:
            with torch.no_grad():
                sl = mmdit.sample_latents((4, *latents.shape[1:]), text[:4], steps=30, device=dev, dtype=dtype)
                imgs = vae.decode_latents(sl.to(vae.dtype))
                save_png(imgs, os.path.join(out_dir, f"sample_step{step}.png"))
            print(f"[overfit] intermediate sample saved @ step {step}", flush=True)

    # sample + save
    print("[overfit] sampling ...", flush=True)
    with torch.no_grad():
        sl = mmdit.sample_latents((4, *latents.shape[1:]), text[:4], steps=30, device=dev, dtype=dtype)
        imgs = vae.decode_latents(sl.to(vae.dtype))
        save_png(imgs, os.path.join(out_dir, "overfit_sample.png"))
    with torch.no_grad():
        ref_imgs = vae.decode_latents(latents[:1].to(vae.dtype))
        save_png(ref_imgs, os.path.join(out_dir, "overfit_reference.png"))
    print(f"[overfit] DONE. final loss ratio = {(loss.item()/init_loss if init_loss else 0):.3f} (target <0.05). samples in {out_dir}", flush=True)
    return {"init_loss": init_loss, "final_loss": loss.item(), "ratio": loss.item() / init_loss if init_loss else 0}
