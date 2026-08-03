#!/bin/bash
# Symlink existing weights into /mnt/data/share/checkpoints root (non-destructive).
set -e
CKPT=/mnt/data/share/checkpoints
mkdir -p "$CKPT/black-forest-labs" "$CKPT/Qwen"
# FLUX.2-dev (exists at workspace path)
[ -e "$CKPT/black-forest-labs/FLUX.2-dev" ] || ln -s /mnt/workspace/users/lzj/Uniworld/models/black-forest-labs/FLUX.2-dev "$CKPT/black-forest-labs/FLUX.2-dev"
echo "checkpoints tree:"; ls -la "$CKPT/black-forest-labs" "$CKPT/Qwen"
