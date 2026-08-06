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
    """Return Flux's log-domain shift ``mu``, linear in image-token count."""
    if max_seq_len == base_seq_len:
        return base_shift
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * max(image_seq_len, 1) + b


def sample_timesteps(
    batch_size: int,
    device,
    strategy: str = "logit_normal",
    image_seq_len: int | None = None,
    shift: float | None = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> torch.Tensor:
    """Sample t in [0,1] (t=0 clean -> t=1 noise). Returns shifted t if applicable."""
    if strategy == "uniform":
        t = torch.rand(batch_size, device=device)
    elif strategy == "logit_normal":
        t = torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_std + logit_mean
        )
    elif strategy == "logit_normal_shift":
        t = torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_std + logit_mean
        )
        if shift is None:
            # calculate_shift follows BFL/Unimm and returns log-domain mu;
            # shift_timesteps expects the positive rational factor alpha.
            mu = calculate_shift(
                image_seq_len or base_seq_len,
                base_seq_len=base_seq_len,
                max_seq_len=max_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            )
            shift = math.exp(mu)
        t = shift_timesteps(t, shift)
    else:
        raise ValueError(f"unknown timestep strategy: {strategy}")
    return t


def timestep_bin_sums(
    timesteps: torch.Tensor,
    fm_loss_per_sample: torch.Tensor,
    self_flow_loss_per_sample: torch.Tensor | None = None,
    num_bins: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return count/FM/Self-Flow sums for equal-width bins on the input device."""
    if timesteps.ndim != 1 or fm_loss_per_sample.shape != timesteps.shape:
        raise ValueError("timesteps and per-sample losses must be matching 1D tensors")
    indices = torch.clamp((timesteps.detach() * num_bins).long(), 0, num_bins - 1)
    counts = torch.zeros(num_bins, device=timesteps.device, dtype=torch.float64)
    fm_sums = torch.zeros_like(counts)
    sf_sums = torch.zeros_like(counts)
    counts.scatter_add_(0, indices, torch.ones_like(timesteps, dtype=torch.float64))
    fm_sums.scatter_add_(0, indices, fm_loss_per_sample.detach().to(torch.float64))
    if self_flow_loss_per_sample is not None:
        sf_sums.scatter_add_(
            0, indices, self_flow_loss_per_sample.detach().to(torch.float64)
        )
    return counts, fm_sums, sf_sums
