"""在 0803-test 的 8 张真实图和真实 caption 上做小模型严格过拟合并输出对照图。"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from PIL import Image, ImageDraw

from uv3.config import load_config
from uv3.data.tar_dataset import TarMetadataDataset
from uv3.modeling.flow import interpolate, velocity_target
from uv3.train.trainer import build_models


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x[:3].clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def make_grid(refs, samples, captions, path):
    n = len(refs)
    cell, label_h = refs[0].width, 46
    canvas = Image.new("RGB", (n * cell, 2 * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (ref, sample, cap) in enumerate(zip(refs, samples, captions)):
        x = i * cell
        canvas.paste(ref, (x, label_h))
        canvas.paste(sample, (x, cell + 2 * label_h))
        draw.text((x + 4, 4), f"#{i} reference", fill="black")
        draw.text((x + 4, cell + label_h + 4), f"#{i} generated", fill="black")
        draw.text((x + 4, 22), cap[:34].replace("\n", " "), fill="black")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/overfit_real_tar.yaml")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sample-every", type=int, default=5000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    torch.manual_seed(cfg.train.seed)
    out = os.path.join(cfg.train.output_dir, cfg.train.run_name)
    os.makedirs(out, exist_ok=True)

    ds = TarMetadataDataset(cfg.data.root, cfg.data.image_size, cfg.data.caption_field, shuffle=False)
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) == args.n:
            break
    if len(rows) != args.n:
        raise RuntimeError(f"只读到 {len(rows)} 张有效真实图")
    pixels = torch.stack([x["pixel_values"] for x in rows]).to(dev, dtype)
    captions = [x["text"] for x in rows]
    vae, qwen, model, opts, _ = build_models(cfg, dev, dtype, dummy_text=False)
    with torch.no_grad():
        latents = vae.encode_images(pixels)
        ids, mask = qwen.tokenize(captions, dev)
        text = qwen.encode_text(ids, mask)
        n_txt = text.shape[1]
        n_img = (latents.shape[-1] // 2) * (latents.shape[-2] // 2)
        text_attn_mask = torch.zeros(args.n, 1, 1, n_txt + n_img, device=dev, dtype=dtype)
        for i in range(args.n):
            valid = int(mask[i].sum().item())
            if valid < n_txt:
                text_attn_mask[i, 0, 0, valid:n_txt] = -65504.0
    refs = [tensor_to_pil(x) for x in pixels]
    for i, im in enumerate(refs):
        im.save(os.path.join(out, f"reference_{i:02d}.png"))
    with open(os.path.join(out, "samples.json"), "w") as f:
        json.dump([{"index": i, "caption": c} for i, c in enumerate(captions)], f, ensure_ascii=False, indent=2)
    del qwen
    torch.cuda.empty_cache()

    for opt in opts.values():
        opt.zero_grad()
    init = None
    g = torch.Generator(device=dev).manual_seed(20260803)
    t0 = time.time()
    last = None
    for step in range(1, args.steps + 1):
        # 每步覆盖全部 8 图；噪声与 t 持续变化，避免只记住固定噪声。
        t = torch.sigmoid(torch.randn(args.n, device=dev, generator=g))
        noise = torch.randn(latents.shape, device=dev, dtype=latents.dtype, generator=g)
        pred = model.predict_velocity(interpolate(latents, noise, t), text, t,
                                      text_attn_mask=text_attn_mask)
        loss = torch.nn.functional.mse_loss(pred.float(), velocity_target(latents, noise).float())
        loss.backward()
        for opt in opts.values():
            opt.step(); opt.zero_grad()
        last = loss.item()
        init = last if init is None else init
        if step == 1 or step % 100 == 0:
            print(f"[real-overfit] step={step} loss={last:.6f} ratio={last/init:.6f} it/s={step/(time.time()-t0):.3f}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            with torch.no_grad():
                torch.manual_seed(20260803)
                sl = model.sample_latents((args.n, *latents.shape[1:]), text, steps=50, device=dev,
                                          dtype=dtype, text_attn_mask=text_attn_mask)
                imgs = vae.decode_latents(sl.to(vae.dtype))
            ims = [tensor_to_pil(x) for x in imgs]
            for i, im in enumerate(ims):
                im.save(os.path.join(out, f"generated_step{step:05d}_{i:02d}.png"))
            make_grid(refs, ims, captions, os.path.join(out, f"comparison_step{step:05d}.png"))
            print(f"[real-overfit] visualization saved step={step}", flush=True)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump({"n": args.n, "steps": args.steps, "init_loss": init, "final_loss": last,
                   "ratio": last/init, "elapsed_seconds": time.time()-t0}, f, indent=2)
    print(f"[real-overfit] DONE ratio={last/init:.6f} out={out}", flush=True)


if __name__ == "__main__":
    main()
