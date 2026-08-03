"""Overfit entry: load config, run cached overfit. CUDA_VISIBLE_DEVICES=5 python -m scripts.run_overfit"""
from __future__ import annotations

import torch

from uv3.config import load_config
from uv3.train.trainer import run_overfit


def main():
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/overfit_tiny.yaml"
    cfg = load_config(cfg_path)
    # bump for a real overfit run (override tiny defaults)
    if cfg.train.max_steps < 500:
        cfg.train.max_steps = 500
    cfg.train.batch_size_per_gpu = 4
    cfg.train.log_every = 100
    setattr(cfg.data, "overfit_n", 8)
    cfg.train.max_steps = 10000
    run_overfit(cfg)


if __name__ == "__main__":
    main()
