#!/bin/bash
set -e
cd /mnt/data/users/wfz/uv3-codebase
SIZE=${1:-1b}
GPUS=${CUDA_VISIBLE_DEVICES:-5,6}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
exec /mnt/data/users/lzj/Uniworld/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=$NGPU --master_port=29512 \
    -m uv3.utils.eff_benchmark --size $SIZE
