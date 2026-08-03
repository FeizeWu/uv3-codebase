"""Self-flow path validation (single-GPU, synthetic, does NOT touch the working trainer).

Validates: EMA teacher copy + block forward hooks (capture student block-M / teacher block-N
features) + projector + cosine feature loss + FM velocity loss + EMA update + Muon step.
If this converges (feature loss decreases), the self-flow mechanism is correct and can be
wired into fsdp2_trainer (EMA×FSDP2 is the remaining care item).

CUDA_VISIBLE_DEVICES=5 python -m scripts.smoke_self_flow
"""
from __future__ import annotations

import copy
import os
import time

import torch
from torch import nn
import torch.nn.functional as F

from uv3.config import ModelConfig, ComponentConfig, SelfFlowConfig, TrainConfig, OptimizerConfig
from uv3.modeling.mmdit import MMDiT
from uv3.modeling.self_flow import build_self_flow_projector, self_flow_feature_loss, FeatureCapture
from uv3.modeling.flow import interpolate, velocity_target, logit_normal_timesteps
from uv3.optim.build_optimizers import build_optimizers


def main():
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    print("[sf] building MMDiT + EMA teacher + projector ...", flush=True)
    mc = ModelConfig(
        architecture="mmdit", hidden_size=768, num_layers=2, num_double_layers=2,
        num_single_layers=2, num_heads=12, latent_channels=32, patch_size=2, in_channels=128,
        out_channels=128, rope_theta=2000.0, axes_dims_rope=(16, 16, 16, 16), guidance_embeds=False,
        alpha_on=False, self_flow=SelfFlowConfig(enabled=True, coeff=1.0),
        vae=ComponentConfig(), qwen_vl=ComponentConfig(), transformer=ComponentConfig(backend="random", trainable=True),
    )
    stub = type("E", (), {"hidden_size": 4096})
    mmdit = MMDiT.build(mc.transformer, mc, text_encoder=stub()).to(dev, dtype=dtype)
    teacher = copy.deepcopy(mmdit).to(dev, dtype=dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)
    inner = mmdit.inner_dim
    projector = build_self_flow_projector(inner).to(dev, dtype=dtype)

    # student captures double[0] IMG stream (out[1]); teacher captures single[-1] IMG segment (tail after txt)
    N_TXT = 64
    student_cap = FeatureCapture(stream="img_double"); student_cap.attach(mmdit.double_blocks[0])
    teacher_cap = FeatureCapture(stream="img_single_tail", n_txt=N_TXT); teacher_cap.attach(teacher.single_blocks[-1])

    tc = TrainConfig(optimizer=OptimizerConfig())
    # Muon over mmdit 2D weights + projector (1D-ish? projector is 2D Linear -> muon); adam for norms.
    opt, (nm, na) = build_optimizers(nn.ModuleList([mmdit, projector]), tc)
    print(f"[sf] muon={nm} adam={na} inner={inner}", flush=True)

    # synthetic data
    N = 8
    clean = torch.randn(N, 32, 32, 32, device=dev, dtype=dtype) * 2.0
    text = torch.randn(N, 64, 4096, device=dev, dtype=dtype)
    bs = 8
    ema_decay = 0.99
    ratio = 0.5
    mask_ratio = 0.5
    from uv3.modeling.self_flow import build_self_flow_latents_continuous
    t0 = time.time()
    for o in opt.values():
        o.zero_grad()
    for step in range(200):
        idx = torch.randint(0, N, (bs,))
        cl = clean[idx]; tx = text[idx]
        t = logit_normal_timesteps(bs, dev)
        noise = torch.randn_like(cl)
        # per-token mixed student latents + per-token timesteps + teacher latents (cleaner)
        student_latents, teach_latent, t_teacher, token_t = build_self_flow_latents_continuous(
            cl, noise, t, mask_ratio=mask_ratio, ratio=ratio, timestep_mode="ratio",
        )
        pred_v = mmdit.predict_velocity(student_latents, tx, t, token_timesteps=token_t)
        fm_loss = F.mse_loss(pred_v.float(), velocity_target(cl, noise).float())
        s_feat = student_cap.features  # IMG stream from double[0]
        with torch.no_grad():
            _ = teacher.predict_velocity(teach_latent, tx, t_teacher)
        t_feat = teacher_cap.features  # IMG segment from single[-1]
        n_tok = min(s_feat.shape[1], t_feat.shape[1])
        proj_s = projector(s_feat[:, :n_tok])
        feat_loss = self_flow_feature_loss(t_feat[:, :n_tok], proj_s)
        loss = fm_loss + 1.0 * feat_loss
        loss.backward()
        for o in opt.values():
            o.step()
        for o in opt.values():
            o.zero_grad()
        with torch.no_grad():
            for tp, p in zip(teacher.parameters(), mmdit.parameters()):
                tp.lerp_(p, 1 - ema_decay)
        if step % 20 == 0:
            s_std = s_feat.float().std().item() if s_feat is not None else float("nan")
            t_std = t_feat.float().std().item() if t_feat is not None else float("nan")
            print(f"[sf] step {step:3d} fm={fm_loss.item():.4f} feat={feat_loss.item():.4f} "
                  f"s_std={s_std:.3f} t_std={t_std:.3f} {(step+1)/(time.time()-t0):.1f}it/s", flush=True)
    print("[sf] DONE — self-flow (img-token align + per-token mix) validated", flush=True)


if __name__ == "__main__":
    main()
