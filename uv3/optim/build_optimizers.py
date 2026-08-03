"""Muon + AdamW parameter split (native torch.optim.Muon, torch>=2.9).

Adapted from UniWorld third_party/DiT/train.py build_optimizers. Differences:
- uses native torch.optim.Muon (no patched build needed; torch 2.12 confirmed)
- predicate matches diffusers Flux2 naming (transformer_blocks./single_transformer_blocks.)
  instead of DiT's `blocks.`
- Muon only accepts 2D params (enforced by the optimizer); the split guarantees it.
- explicit args: UniWorld defaults (muon lr=1e-4, wd=0.0, mom=0.95, nesterov, ns=5,
  adjust_lr_fn='match_rms_adamw'); do NOT eat torch defaults (1e-3/0.1).
"""
from __future__ import annotations

import torch


def _is_muon_param(name: str, param: torch.nn.Parameter) -> bool:
    """2D weight matrices inside MMDiT transformer blocks -> Muon."""
    if param.ndim != 2 or not param.requires_grad:
        return False
    return "transformer_blocks." in name or "single_transformer_blocks." in name


def build_optimizers(model, cfg):
    """Return ({'muon':..., 'adam':...}, (n_muon_tensors, n_adam_tensors))."""
    o = cfg.optimizer if hasattr(cfg, "optimizer") else cfg.train.optimizer
    # Preserve PyTorch's default fused=None/foreach auto-selection unless the
    # fused path is explicitly requested.  Passing fused=False disables that
    # fast default on CUDA in torch 2.10.
    adam_impl_kwargs = {"fused": True} if bool(getattr(o, "adam_fused", False)) else {}
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if o.optimizer == "adamw":
        if not trainable:
            raise RuntimeError("AdamW received no trainable parameters")
        return {
            "adam": torch.optim.AdamW(
                [p for _, p in trainable],
                lr=o.adam_lr,
                betas=tuple(o.adam_betas),
                eps=o.adam_eps,
                weight_decay=o.adam_weight_decay,
                **adam_impl_kwargs,
            )
        }, (0, len(trainable))
    if o.optimizer != "muon_adam":
        raise ValueError(f"unsupported optimizer: {o.optimizer}")

    muon_named = [(n, p) for n, p in model.named_parameters() if _is_muon_param(n, p)]
    muon_ids = {id(p) for _, p in muon_named}
    adam_named = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in muon_ids
    ]
    grouped = muon_ids | {id(p) for _, p in adam_named}
    if len(grouped) != len(trainable):
        raise RuntimeError("Muon/Adam groups overlap or miss trainable params")
    if not muon_named or not adam_named:
        raise RuntimeError("both Muon and Adam groups must be non-empty")

    optimizers = {
        "muon": torch.optim.Muon(
            [p for _, p in muon_named],
            lr=o.muon_lr,
            momentum=o.muon_momentum,
            nesterov=o.muon_nesterov,
            ns_steps=o.muon_ns_steps,
            weight_decay=o.muon_weight_decay,
            adjust_lr_fn="match_rms_adamw" if o.muon_rms_scale else None,
        ),
        "adam": torch.optim.AdamW(
            [p for _, p in adam_named],
            lr=o.adam_lr,
            betas=tuple(o.adam_betas),
            eps=o.adam_eps,
            weight_decay=o.adam_weight_decay,
            **adam_impl_kwargs,
        ),
    }
    return optimizers, (len(muon_named), len(adam_named))
