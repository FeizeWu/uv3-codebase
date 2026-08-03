#!/bin/bash
set -e
cd /mnt/data/users/wfz/uv3-codebase
CFG=${1:-configs/train.yaml}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
exec /mnt/data/users/lzj/Uniworld/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=$NGPU --master_port=29550 \
    -m uv3.train.fsdp2_trainer --config "$CFG"
