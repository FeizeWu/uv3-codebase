#!/usr/bin/env bash
set -euo pipefail

# Keep the five Qwen/MMDiT static graphs off both the small system /tmp and the
# metadata-bound shared CPFS. Override this for a prepared node-local cache.
export TORCHINDUCTOR_CACHE_DIR="${UV3_COMPILE_CACHE_DIR:-/dev/shm/uv3-inductor-cache}"
# 8 ranks x torch's default 32 workers oversubscribes this 184-CPU host.
export TORCHINDUCTOR_COMPILE_THREADS="${UV3_COMPILE_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec .venv-cu128/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m uv3.train.fsdp2_trainer \
  --config configs/bench_real_3b_8gpu_adamw_no_sf_qwen_mmdit_5buckets_best.yaml
