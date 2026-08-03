"""Rectified-flow math. Convention: t=0 is CLEAN data, t=1 is NOISE.

velocity target = noise - clean  (dx/dt of interpolate; v3 correction — NOT clean-noise).
Reuses UniWorld uniworld/modeling/flow.py verbatim.
"""
from __future__ import annotations

import torch


def shift_timesteps(timesteps: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    if shift <= 0:
        raise ValueError(f"timestep shift must be positive, got {shift}")
    return shift * timesteps / (1 + (shift - 1) * timesteps)


def logit_normal_timesteps(batch_size: int, device, shift: float = 1.0) -> torch.Tensor:
    timesteps = torch.sigmoid(torch.randn(batch_size, device=device))
    return shift_timesteps(timesteps, shift)


def interpolate(clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    timesteps = timesteps.reshape(-1, *((1,) * (clean.ndim - 1)))
    return (1 - timesteps) * clean + timesteps * noise


def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # dx/dt of interpolate: d/dt[(1-t)clean + t noise] = noise - clean
    return noise - clean


def euler_schedule(steps: int, device, dtype=torch.float32, shift: float = 1.0) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"sampling steps must be positive, got {steps}")
    base = torch.linspace(1, 0, steps + 1, device=device, dtype=dtype)
    return shift_timesteps(base, shift)


def euler_step(sample: torch.Tensor, velocity: torch.Tensor, current, following) -> torch.Tensor:
    # x_{t-dt} = x_t + v * (t_next - t) ; t decreases 1->0
    return sample + velocity * (following - current)
