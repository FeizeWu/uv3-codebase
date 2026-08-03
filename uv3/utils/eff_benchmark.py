"""Efficiency benchmark: measure MMDiT throughput at 1B/3B/7B, report machines/time for 1B images.

Codebase口径 (default): dummy latents + pre-computed-dim text feats -> pure MMDiT fwd+bwd+Muon.
Real口径 (optional --real): includes VAE encode + Qwen3.5 fwd per step.
FLOPS: total = 8 × N_mmdit × T_tokens × 1e9  (8 = 6 student fwd+bwd + 2 teacher no-grad fwd;
no extra overhead mult — v3 correction). N_mmdit = MMDiT params only; T = img+txt tokens.
H20 BF16 dense = 148 TFLOPS (MFU denominator).
Run: torchrun --nproc_per_node=2 scripts/eff_benchmark.py --size 7b
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from ..config import load_config
from ..modeling.mmdit import MMDiT
from ..optim.build_optimizers import build_optimizers
from ..train.fsdp2 import apply_fsdp2, make_mesh

H20_BF16_TFLOPS = 148.0


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["1b", "3b", "7b"], default="1b")
    ap.add_argument("--config", default=None, help="override config path")
    ap.add_argument("--real", action="store_true", help="include VAE+Qwen fwd (real口径)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=0, help="override batch_per_gpu")
    args = ap.parse_args()

    cfg_path = args.config or f"configs/eff_{args.size}.yaml"
    cfg = load_config(cfg_path)
    if args.batch:
        cfg.train.batch_size_per_gpu = args.batch
    if not cfg.train.max_steps:
        cfg.train.max_steps = args.steps

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = 0
    dev = torch.device("cuda", rank)
    dtype = torch.bfloat16

    # build MMDiT only (codebase口径: dummy text = randn in joint_attention_dim)
    mc = cfg.model
    text_dim = 4096
    qwen = None
    if args.real:
        from ..modeling.qwen3_5 import Qwen3_5TextEncoder
        qwen = Qwen3_5TextEncoder.from_pretrained(mc.qwen_vl.pretrained, max_length=mc.qwen_vl.max_length, dtype=dtype).to(dev).eval()
        text_dim = qwen.hidden_size
    class _EncStub:
        hidden_size = text_dim
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=(qwen or _EncStub())).to(dev, dtype=dtype)
    if os.environ.get("UV3_COMPILE"):
        mmdit.transformer = torch.compile(mmdit.transformer)  # AC -> compile -> fully_shard (torchtitan order)
        if rank == 0:
            print("[eff] torch.compile ON (mmdit.transformer, pre-fsdp2)", flush=True)

    mesh = None
    if distributed:
        mesh = make_mesh(num_replicate=cfg.train.num_replicate)
        apply_fsdp2(mmdit, mesh, reshard_after_forward=cfg.train.reshard_after_forward)

    optimizers, (n_m, n_a) = build_optimizers(mmdit, cfg)
    n_params = count_params(mmdit)
    bs = cfg.train.batch_size_per_gpu
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        print(f"[eff:{args.size}] MMDiT params={n_params:,} ({n_params/1e9:.2f}B) bs={bs} world={world} muon={n_m} adam={n_a} real={args.real}", flush=True)

    # tokens per image: img 256 + txt 64 = 320 (256px latent 32x32 packed 16x16)
    T_tokens = 256 + 64
    H, W = 32, 32
    _cap = ["a photo of a cat, ultra hd, cinematic"] * bs

    def make_text():
        if qwen is not None:
            with torch.no_grad():
                ids, mask = qwen.tokenize(_cap, dev)
                return qwen.encode_text(ids, mask)
        return torch.randn(bs, 64, text_dim, device=dev, dtype=dtype)

    # warmup
    for o in optimizers.values():
        o.zero_grad()
    for _ in range(args.warmup):
        clean = torch.randn(bs, 32, H, W, device=dev, dtype=dtype)
        text = make_text()
        t = torch.sigmoid(torch.randn(bs, device=dev))
        noise = torch.randn_like(clean)
        from ..modeling.flow import interpolate, velocity_target
        noisy = interpolate(clean, noise, t)
        v = mmdit.predict_velocity(noisy, text, t)
        loss = torch.nn.functional.mse_loss(v.float(), velocity_target(clean, noise).float())
        loss.backward()
        for o in optimizers.values():
            o.step()
        for o in optimizers.values():
            o.zero_grad()
    torch.cuda.synchronize()

    # timed
    t0 = time.time()
    peak0 = torch.cuda.max_memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    n = args.steps
    for _ in range(n):
        clean = torch.randn(bs, 32, H, W, device=dev, dtype=dtype)
        text = make_text()
        t = torch.sigmoid(torch.randn(bs, device=dev))
        noise = torch.randn_like(clean)
        noisy = interpolate(clean, noise, t)
        v = mmdit.predict_velocity(noisy, text, t)
        loss = torch.nn.functional.mse_loss(v.float(), velocity_target(clean, noise).float())
        loss.backward()
        for o in optimizers.values():
            o.step()
        for o in optimizers.values():
            o.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1e9

    # bs is PER-GPU; GPUs run lockstep -> n*bs/elapsed is the PER-GPU rate.
    per_gpu_sps = n * bs / elapsed
    samples_per_sec = per_gpu_sps * world            # total across all GPUs
    # 6 = 2 fwd + 4 bwd (BASIC FM). bs is PER-GPU batch -> total_flops is per-GPU already.
    total_flops_per_gpu = 6 * n_params * T_tokens * n * bs
    achieved_tflops = total_flops_per_gpu / elapsed / 1e12  # per-GPU TFLOPS
    mfu = achieved_tflops / H20_BF16_TFLOPS
    time_1b_1node_h = 1e9 / (samples_per_sec) / 3600  # hours on `world` gpus
    # per-GPU throughput extrapolated to 8-GPU node (regardless of how many we measured on)
    per_gpu_8gpu_throughput = per_gpu_sps * 8
    days_1node_8gpu = 1e9 / (per_gpu_8gpu_throughput * 86400)  # days on 1 node (8 GPU)
    # nodes (8 gpus each) for 7d / 30d target wall-clock — derived from per-GPU throughput
    sec_7d = 7 * 24 * 3600
    sec_30d = 30 * 24 * 3600
    nodes_7d = (1e9 / (per_gpu_8gpu_throughput * sec_7d)) if per_gpu_sps > 0 else float("inf")
    nodes_30d = (1e9 / (per_gpu_8gpu_throughput * sec_30d)) if per_gpu_sps > 0 else float("inf")

    if rank == 0:
        print("=" * 70, flush=True)
        print(f"[eff:{args.size}] REPORT (codebase口径, {'FSDP2' if distributed else 'single-GPU'})", flush=True)
        print(f"  MMDiT params        : {n_params:,} ({n_params/1e9:.2f} B)", flush=True)
        print(f"  GPUs / batch-per-gpu: {world} / {bs}", flush=True)
        print(f"  tokens/image       : {T_tokens} (img 256 + txt 64)", flush=True)
        print(f"  samples/sec (total) : {samples_per_sec:.2f}  (per-GPU {per_gpu_sps:.2f})", flush=True)
        print(f"  per-GPU TFLOPS      : {achieved_tflops:.1f} (MFU={mfu*100:.1f}% vs {H20_BF16_TFLOPS}T)", flush=True)
        print(f"  peak mem/GPU        : {peak:.1f} GB", flush=True)
        print(f"  time/step           : {elapsed/n*1000:.1f} ms", flush=True)
        print(f"  ---- extrapolation to 1e9 images ----", flush=True)
        print(f"  on {world} GPU: {time_1b_1node_h:.1f} hours = {time_1b_1node_h/24:.1f} days", flush=True)
        print(f"  on 8 GPU (1 node): {days_1node_8gpu:.1f} days", flush=True)
        print(f"  nodes (8 GPU each) for  7 days: {nodes_7d:.2f}", flush=True)
        print(f"  nodes (8 GPU each) for 30 days: {nodes_30d:.2f}", flush=True)
        print(f"  FLOPS check: per-GPU flops={total_flops_per_gpu:.3e} achieved={achieved_tflops:.1f}T/s MFU={mfu*100:.1f}%", flush=True)
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
