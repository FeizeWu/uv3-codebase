#!/bin/bash
set -e
cd /mnt/data/users/wfz/uv3-codebase
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export PYTHONPATH=.
exec /mnt/data/users/lzj/Uniworld/.venv/bin/python scripts/run_overfit.py uv3/configs/overfit_tiny.yaml
