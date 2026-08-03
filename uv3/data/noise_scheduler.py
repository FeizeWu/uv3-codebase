"""Timestep sampling for flow matching. Self-written (NOT reusing transfusion-core,
whose t convention is inverted: t=0 noise / t=1 clean). Here t=0 CLEAN, t=1 NOISE —
matching flow.py, with Flux-style resolution shift (calculate_shift) but NO double-flip.
"""
from __future__ import annotations

import math

import torch

from ..modeling.flow import shift_timesteps


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    """Flux resolution-dependent shift: linear in log(seq_len) between base/max."""
    if max_seq_len == base_seq_len:
        return base_shift
    m = (max_shift - base_shift) / (math.log(max_seq_len) - math.log(base_seq_len))
    b = base_shift - m * math.log(base_seq_len)
    return m * math.log(max(image_seq_len, 1)) + b


def sample_timesteps(
    batch_size: int,
    device,
    strategy: str = "logit_normal",
    image_seq_len: int | None = None,
    shift: float | None = None,
) -> torch.Tensor:
    """Sample t in [0,1] (t=0 clean -> t=1 noise). Returns shifted t if applicable."""
    if strategy == "uniform":
        t = torch.rand(batch_size, device=device)
    elif strategy == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    elif strategy == "logit_normal_shift":
        t = torch.sigmoid(torch.randn(batch_size, device=device))
        if shift is None:
            shift = calculate_shift(image_seq_len) if image_seq_len else 1.0
        t = shift_timesteps(t, shift)
    else:
        raise ValueError(f"unknown timestep strategy: {strategy}")
    return t
