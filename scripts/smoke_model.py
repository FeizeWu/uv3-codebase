"""Minimal model smoke: VAE + Qwen3.5 + MMDiT, 1 forward+backward, basic FM loss.

Validates vae.py (BN), qwen3_5.py (text encoder+bridge), mmdit.py (Flux2 dual/single),
flow.py (noise-clean target), build_optimizers.py (native Muon split). No FSDP2, no data
pipeline, no self-flow. Run: CUDA_VISIBLE_DEVICES=5 python -m scripts.smoke_model
"""
from __future__ import annotations

import os

import torch
from torch import nn

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig, TrainConfig, OptimizerConfig
from uv3.modeling.vae import Flux2VAE
from uv3.modeling.qwen3_5 import Qwen3_5TextEncoder, Qwen3_5EmbeddingBridge
from uv3.modeling.mmdit import MMDiT
from uv3.optim.build_optimizers import build_optimizers

FLUX = "/mnt/data/share/checkpoints/black-forest-labs/FLUX.2-dev"
QWEN = "/mnt/data/share/checkpoints/Qwen/Qwen3.5-9B"


def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        architecture="mmdit", hidden_size=768,
        num_layers=2, num_double_layers=2, num_single_layers=2, num_heads=12,
        latent_channels=32, patch_size=2, in_channels=128, out_channels=128,
        rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16), guidance_embeds=False,
        alpha_on=False, self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(), transformer=ComponentConfig(backend="random", trainable=True),
    )


def tiny_train_cfg() -> TrainConfig:
    return TrainConfig(optimizer=OptimizerConfig())


class Bundle(nn.Module):
    """Holds MMDiT so build_optimizers sees named params (transformer_blocks.)."""

    def __init__(self, mmdit):
        super().__init__()
        self.mmdit = mmdit


def main():
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    print("[smoke] loading VAE ...", flush=True)
    vae = Flux2VAE.from_pretrained(FLUX, dtype=dtype).to(dev).eval()

    print("[smoke] loading Qwen3.5-9B text encoder ...", flush=True)
    qwen = Qwen3_5TextEncoder.from_pretrained(QWEN, max_length=64, dtype=dtype).to(dev).eval()

    print("[smoke] building tiny MMDiT ...", flush=True)
    mc = tiny_model_cfg()
    if os.environ.get("UV3_FLEX"):
        mc.flex_attention = True
        print("[smoke] flex_attention ON", flush=True)
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=qwen).to(dev, dtype=dtype)
    bundle = Bundle(mmdit).to(dev, dtype=dtype)

    n_params = sum(p.numel() for p in bundle.parameters() if p.requires_grad)
    print(f"[smoke] trainable params: {n_params:,}", flush=True)

    tc = tiny_train_cfg()
    optimizers, (n_m, n_a) = build_optimizers(bundle, tc)
    print(f"[smoke] optimizers: muon tensors={n_m} adam tensors={n_a}", flush=True)

    # fake batch: 2 images 256x256, captions
    imgs = torch.rand(2, 3, 256, 256, device=dev, dtype=dtype)
    print("[smoke] encoding images ...", flush=True)
    latents = vae.encode_images(imgs)
    print(f"[smoke] latent shape: {tuple(latents.shape)}", flush=True)

    captions = ["a photo of a cat", "a photo of a dog"]
    ids, mask = qwen.tokenize(captions, dev)
    print(f"[smoke] token ids: {tuple(ids.shape)} pad_mask sum: {mask.sum().item()}", flush=True)
    # pass RAW 4096-dim Qwen features; Flux2's internal context_embedder projects 4096->inner_dim
    text = qwen.encode_text(ids, mask)
    print(f"[smoke] text feats: {tuple(text.shape)} (qwen hidden={qwen.hidden_size}, mmdit inner={mmdit.inner_dim})", flush=True)

    print("[smoke] training_loss (basic FM) ...", flush=True)
    loss = mmdit.training_loss(latents.to(dtype), text.to(dtype))
    print(f"[smoke] loss = {loss.item():.6f}", flush=True)

    for o in optimizers.values():
        o.zero_grad()
    loss.backward()
    for o in optimizers.values():
        o.step()
    print("[smoke] backward+step OK", flush=True)

    # it2i ref path (num_ref_tokens>0): 256px -> latent 32x32 -> packed 16x16 = 256 ref tokens
    print("[smoke] it2i ref path ...", flush=True)
    ref = vae.encode_images(imgs.flip(0))
    n_ref = (ref.shape[-1] // 2) * (ref.shape[-2] // 2)   # 16*16 = 256
    loss2 = mmdit.training_loss(latents.to(dtype), text.to(dtype), ref=ref.to(dtype), num_ref_tokens=n_ref)
    print(f"[smoke] it2i loss = {loss2.item():.6f} (num_ref_tokens={n_ref}, ref path ran)", flush=True)

    print("[smoke] DONE — model forward/backward validated", flush=True)


if __name__ == "__main__":
    main()
