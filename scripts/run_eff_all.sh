#!/bin/bash
# Run efficiency benchmark for 1b/3b/7b and print the report table.
# Usage: CUDA_VISIBLE_DEVICES=0 scripts/run_eff_all.sh   (single-GPU 1b/3b; 7b needs 2 GPUs)
set -e
cd /mnt/data/users/wfz/uv3-codebase
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/data/users/lzj/Uniworld/.venv/bin/python
GPUS=${CUDA_VISIBLE_DEVICES:-0}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l | tr -d ' ')
echo "===== 1B (single GPU, batch16) ====="
CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f1) $PY -m uv3.utils.eff_benchmark --size 1b --batch 16 2>&1 | tail -14
echo "===== 3B (single GPU, batch8) ====="
CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f1) $PY -m uv3.utils.eff_benchmark --size 3b --batch 8 2>&1 | tail -14
if [ "$NGPU" -ge 2 ]; then
  echo "===== 7B (2-GPU FSDP2, batch16) ====="
  $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29560 -m uv3.utils.eff_benchmark --size 7b --batch 16 --warmup 10 --steps 40 2>&1 | tail -14
fi
echo "===== real口径 (with Qwen3.5 encoder) ====="
echo "1B real:"; CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f1) $PY -m uv3.utils.eff_benchmark --size 1b --batch 16 --real --warmup 5 --steps 30 2>&1 | grep -E "samples/sec|MFU|nodes" | tail -4
echo "3B real:"; CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f1) $PY -m uv3.utils.eff_benchmark --size 3b --batch 8 --real --warmup 5 --steps 30 2>&1 | grep -E "samples/sec|MFU|nodes" | tail -4
