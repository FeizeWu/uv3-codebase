#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${UV3_REPO_ROOT:-/mnt/data/users/wfz/uv3-codebase}"
PYTHON_BIN="${UV3_PYTHON_BIN:-${REPO_ROOT}/.venv-cu128/bin/python}"
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/eval_4b_test.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29685}"

: "${RUN_NAME:?set RUN_NAME to the training run to evaluate}"
RUN_DIR="${RUN_DIR:-/mnt/oss/users/wfz/uv3-codebase-runs/${RUN_NAME}}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${RUN_DIR}/config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/ckpt.pt}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_HOME="${TORCH_HOME:-/mnt/data/users/wfz/checkpoints/torch-cache}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-3}"

INCEPTION_WEIGHTS="${TORCH_HOME}/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
for required_path in \
  "${PYTHON_BIN}" \
  "${TRAIN_CONFIG}" \
  "${EVAL_CONFIG}" \
  "${CHECKPOINT}" \
  "${INCEPTION_WEIGHTS}" \
  "/mnt/data/share/checkpoints/openai-mirror/clip-vit-base-patch32"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[error] missing required path: ${required_path}" >&2
    exit 1
  fi
done
inception_sha256="$(sha256sum "${INCEPTION_WEIGHTS}" | awk '{print $1}')"
if [[ "${inception_sha256}" != "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2" ]]; then
  echo "[error] invalid Inception weight: ${INCEPTION_WEIGHTS} sha256=${inception_sha256}" >&2
  exit 1
fi
if [[ "${GPUS_PER_NODE}" != "8" ]]; then
  echo "[error] evaluation config expects an 8-way FSDP mesh" >&2
  exit 1
fi
if [[ "${REQUIRE_IDLE_GPUS:-1}" == "1" ]] && \
   [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "[error] GPU compute processes are present; evaluation must not compete with training" >&2
  exit 1
fi

extra_args=(--student)
if [[ -n "${MAX_FIXED:-}" ]]; then extra_args+=(--max-fixed "${MAX_FIXED}"); fi
if [[ -n "${MAX_GENERATED:-}" ]]; then extra_args+=(--max-generated "${MAX_GENERATED}"); fi
if [[ "${SKIP_DISTRIBUTION:-0}" == "1" ]]; then extra_args+=(--skip-distribution); fi
if [[ "${SKIP_FIXED:-0}" == "1" ]]; then extra_args+=(--skip-fixed); fi
if [[ -n "${SAMPLE_STEPS:-}" ]]; then extra_args+=(--sample-steps "${SAMPLE_STEPS}"); fi

echo "[eval] weights=student run=${RUN_NAME} checkpoint=${CHECKPOINT}"
echo "[eval] fixed=train+heldout; distribution=heldout FID/KID/CLIPScore"

# The caller may itself be a multi-node launcher. Remove inherited elastic and
# rank variables, then use an explicit one-node static rendezvous. Otherwise a
# completed 4-node job can make this new local torchrun wait for departed nodes.
exec env \
  -u RANK \
  -u WORLD_SIZE \
  -u LOCAL_RANK \
  -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK \
  -u ROLE_RANK \
  -u ROLE_WORLD_SIZE \
  -u MASTER_ADDR \
  -u MASTER_PORT \
  -u TORCHELASTIC_RUN_ID \
  -u TORCHELASTIC_RESTART_COUNT \
  -u TORCHELASTIC_MAX_RESTARTS \
  -u TORCHELASTIC_ERROR_FILE \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  -m uv3.eval.checkpoint_evaluator \
  --train-config "${TRAIN_CONFIG}" \
  --eval-config "${EVAL_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --run-dir "${RUN_DIR}" \
  "${extra_args[@]}"
