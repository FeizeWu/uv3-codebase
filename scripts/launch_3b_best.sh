#!/usr/bin/env bash
set -euo pipefail

# Keep the five Qwen/MMDiT static graphs off the small system /tmp partition.
export TORCHINDUCTOR_CACHE_DIR="${UV3_COMPILE_CACHE_DIR:-/mnt/data/users/wfz/torchinductor-cache-uv3}"
# 8 ranks x torch's default 32 workers oversubscribes this 184-CPU host.
export TORCHINDUCTOR_COMPILE_THREADS="${UV3_COMPILE_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec .venv-cu128/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m uv3.train.fsdp2_trainer \
  --config configs/bench_real_3b_8gpu_adamw_no_sf_qwen_mmdit_5buckets_best.yaml
