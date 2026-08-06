#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${UV3_REPO_ROOT:-/mnt/data/users/wfz/uv3-codebase}"
PYTHON_BIN="${UV3_PYTHON_BIN:-${REPO_ROOT}/.venv-cu128/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train_pure_single_4b_self_flow_4node.yaml}"
MANIFEST="${MANIFEST:-/mnt/oss/uv3-pretrain-manifect/0803-test/eval-v1/train_manifest.jsonl}"

cd "${REPO_ROOT}"

export NNODES="${NNODES:-${WORLD_SIZE:-4}}"
export NODE_RANK="${NODE_RANK:-${RANK:-}}"
export MASTER_ADDR="${MASTER_ADDR:-}"
export MASTER_PORT="${MASTER_PORT:-29675}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Resume the completed 10k run by default. MAX_STEPS is the final total step,
# not the number of extra steps, so 20000 continues for another 10000 steps.
# These defaults are identical on every node; callers may override MAX_STEPS.
RUN_NAME="${RUN_NAME:-train_pure_single_4b_self_flow_4node_10k}"
MAX_STEPS="${MAX_STEPS:-20000}"
ALLOW_RESUME="${ALLOW_RESUME:-1}"
RESUME_EXPECTED_STEP="${RESUME_EXPECTED_STEP:-10000}"

if [[ "${NNODES}" != "4" ]]; then
  echo "[error] this launcher is fixed for 4 nodes; got NNODES=${NNODES}" >&2
  exit 1
fi
if [[ -z "${NODE_RANK}" || ! "${NODE_RANK}" =~ ^[0-3]$ ]]; then
  echo "[error] NODE_RANK (or scheduler RANK) must be one of 0,1,2,3" >&2
  exit 1
fi
if [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "127.0.0.1" ]]; then
  echo "[error] MASTER_ADDR must be node-rank-0's reachable IP/hostname" >&2
  exit 1
fi
if [[ "${GPUS_PER_NODE}" != "8" ]]; then
  echo "[error] UV3 config uses node-local num_shard=8; got GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 1
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] MAX_STEPS must be a positive integer; got ${MAX_STEPS}" >&2
  exit 1
fi
if [[ "${ALLOW_RESUME}" == "1" ]] && (( MAX_STEPS <= RESUME_EXPECTED_STEP )); then
  echo "[error] MAX_STEPS=${MAX_STEPS} must be greater than resumed step ${RESUME_EXPECTED_STEP}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/uv3-inductor-${RUN_NAME}}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export UV3_CKPT_STAGING_DIR="${UV3_CKPT_STAGING_DIR:-/mnt/data/users/wfz/uv3-checkpoint-staging}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-136}"
export NCCL_IB_SL="${NCCL_IB_SL:-5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"

export UV3_RUN_NAME="${RUN_NAME}"
export UV3_MAX_STEPS="${MAX_STEPS}"
export UV3_MONITOR="${UV3_MONITOR:-1}"
export UV3_MONITOR_NAME="${UV3_MONITOR_NAME:-${RUN_NAME}}"
export CONFIG_PATH="${CONFIG}"

MONITOR_ROOT="${UV3_MONITOR_ROOT:-/mnt/data/users/wfz/uv3-training-monitor}"
MONITOR_PYTHON="${UV3_MONITOR_PYTHON:-${MONITOR_ROOT}/backend/.venv/bin/python}"
telemetry_pid=""
train_pid=""
eval_pid=""

stop_child() {
  local child_pid="${1:-}"
  if [[ "${child_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_child "${eval_pid}"
  stop_child "${train_pid}"
  stop_child "${telemetry_pid}"
}
trap cleanup EXIT INT TERM

for required_path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${MANIFEST}" \
  "/mnt/data/users/wfz/checkpoints/black-forest-labs/FLUX.2-dev/vae" \
  "/mnt/data/share/checkpoints/Qwen/Qwen3.5-9B" \
  "/mnt/data/share/checkpoints/openai-mirror/clip-vit-base-patch32" \
  "/mnt/data/users/wfz/checkpoints/torch-cache/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[error] missing required path: ${required_path}" >&2
    exit 1
  fi
done

visible_gpu_count="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
if [[ "${visible_gpu_count}" != "8" ]]; then
  echo "[error] CUDA_VISIBLE_DEVICES must contain exactly 8 GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "${REQUIRE_IDLE_GPUS:-1}" == "1" ]] && \
   [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "[error] this node already has GPU compute processes" >&2
  nvidia-smi >&2
  exit 1
fi

export UV3_LAUNCH_MANIFEST="${MANIFEST}"
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from uv3.config import load_config
from uv3.train.fsdp2 import resolve_mesh_shape

cfg = load_config(os.environ["CONFIG_PATH"])
manifest = Path(os.environ["UV3_LAUNCH_MANIFEST"])
errors = []
if cfg.data.dataset != "tar": errors.append(f"data.dataset={cfg.data.dataset!r}")
if Path(cfg.data.root) != manifest: errors.append(f"config data.root={cfg.data.root!r}")
expected_vae = Path("/mnt/data/users/wfz/checkpoints/black-forest-labs/FLUX.2-dev")
if Path(cfg.model.vae.pretrained) != expected_vae:
    errors.append(f"config vae.pretrained={cfg.model.vae.pretrained!r}")
if cfg.data.caption_field != "caption_qwen3_7_flash": errors.append(f"caption_field={cfg.data.caption_field!r}")
expected_aspects = ("square", "landscape", "portrait", "widescreen", "phone")
if not cfg.data.bucket: errors.append("resolution bucketing is disabled")
if tuple(cfg.data.aspect_buckets) != expected_aspects:
    errors.append(f"aspect_buckets={cfg.data.aspect_buckets!r}")
if not cfg.data.online_joint_bucketing:
    errors.append("online_joint_bucketing is disabled")
if tuple(cfg.train.text_length_buckets) != (512, 640, 768, 896, 1024):
    errors.append(f"text_length_buckets={cfg.train.text_length_buckets!r}")
if cfg.train.pad_text_to_max_length:
    errors.append("pad_text_to_max_length must be false for joint buckets")
if cfg.model.num_double_layers != 0 or cfg.model.num_single_layers != 30:
    errors.append(f"layers={cfg.model.num_double_layers}+{cfg.model.num_single_layers}")
if not cfg.model.self_flow.enabled: errors.append("Self-Flow is disabled")
sf = cfg.model.self_flow
if sf.timestep_mode != "independent": errors.append(f"self_flow.timestep_mode={sf.timestep_mode!r}")
if sf.mask_ratio != 0.25: errors.append(f"self_flow.mask_ratio={sf.mask_ratio!r}")
if sf.coeff != 0.8: errors.append(f"self_flow.coeff={sf.coeff!r}")
if sf.student_depth_ratio != 0.3 or sf.teacher_depth_ratio != 0.7:
    errors.append(
        f"self_flow.depth_ratios={sf.student_depth_ratio!r}->{sf.teacher_depth_ratio!r}"
    )
if cfg.train.timestep_strategy != "logit_normal_shift" or cfg.train.timestep_shift != 4.63:
    errors.append(
        f"timestep={cfg.train.timestep_strategy!r}/shift={cfg.train.timestep_shift!r}"
    )
if cfg.train.compile_vae and cfg.train.vae_compile_mode != "max-autotune-no-cudagraphs":
    errors.append(f"unsafe vae_compile_mode={cfg.train.vae_compile_mode!r}")
if cfg.train.num_shard != 8: errors.append(f"num_shard={cfg.train.num_shard!r}")
if not cfg.train.fsdp2: errors.append("FSDP2 is disabled")
if errors:
    raise SystemExit("[error] config preflight failed: " + "; ".join(errors))
print(
    "[preflight] config OK",
    f"mesh={resolve_mesh_shape(32, cfg.train.num_replicate, cfg.train.num_shard)}",
    f"optimizer={cfg.train.optimizer.optimizer}",
    f"batch_per_gpu={cfg.train.batch_size_per_gpu}",
    f"global_batch={cfg.train.batch_size_per_gpu * 32 * cfg.train.grad_accum}",
    f"manifest_bytes={manifest.stat().st_size}",
)
PY

output_dir="$("${PYTHON_BIN}" -c 'from uv3.config import load_config; import os; c=load_config(os.environ["CONFIG_PATH"]); print(c.train.output_dir)')/${RUN_NAME}"
checkpoint="${output_dir}/ckpt.pt"
if [[ "${ALLOW_RESUME}" == "1" ]]; then
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] ALLOW_RESUME=1 but checkpoint is missing: ${checkpoint}" >&2
    exit 1
  fi
else
  if [[ -e "${checkpoint}" ]]; then
    echo "[error] checkpoint already exists; set ALLOW_RESUME=1 only for an intentional resume: ${checkpoint}" >&2
    exit 1
  fi
fi

mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${UV3_CKPT_STAGING_DIR}"
log_dir="${output_dir}/launcher_logs/node${NODE_RANK}"
mkdir -p "${log_dir}"

if [[ "${UV3_MONITOR}" == "1" ]]; then
  for monitor_path in "${MONITOR_PYTHON}" "${MONITOR_ROOT}/backend/collect_gpu_telemetry.py"; do
    if [[ ! -e "${monitor_path}" ]]; then
      echo "[error] missing monitor path: ${monitor_path}" >&2
      exit 1
    fi
  done
fi

code_fingerprint="$(sha256sum uv3/train/fsdp2_trainer.py uv3/train/fsdp2.py uv3/data/tar_dataset.py uv3/modeling/mmdit.py "${CONFIG}" | sha256sum | awk '{print $1}')"
echo "[launch] node=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} gpus/node=${GPUS_PER_NODE}"
echo "[launch] python=${PYTHON_BIN} config=${CONFIG} run=${RUN_NAME} max_steps=${MAX_STEPS}"
echo "[launch] manifest=${MANIFEST} code_fingerprint=${code_fingerprint}"
echo "[launch] output=${output_dir} resume=${ALLOW_RESUME} expected_step=${RESUME_EXPECTED_STEP}"

if [[ "${UV3_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[preflight] complete; UV3_PREFLIGHT_ONLY=1, not launching"
  exit 0
fi

if [[ "${UV3_MONITOR}" == "1" && "${NODE_RANK}" == "0" ]]; then
  UV3_MONITOR_RUN_DIR="${output_dir}" UV3_MONITOR_DISPLAY_NAME="${UV3_MONITOR_NAME}" \
  UV3_MONITOR_MAX_STEPS="${MAX_STEPS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["UV3_MONITOR_RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
target = run_dir / "run.json"
temporary = run_dir / "run.json.tmp"
temporary.write_text(
    json.dumps(
        {
            "monitor_enabled": True,
            "display_name": os.environ["UV3_MONITOR_DISPLAY_NAME"],
            "train": {"max_steps": int(os.environ["UV3_MONITOR_MAX_STEPS"])},
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(target)
PY
  echo "[monitor] run registered in shared storage; public UI starts separately on demand"
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --log-dir "${log_dir}" \
    --tee 3 \
    -m uv3.train.fsdp2_trainer \
    --config "${CONFIG}" \
    > >(tee -a "${output_dir}/stdout.log") 2>&1 &
else
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --log-dir "${log_dir}" \
  --tee 3 \
  -m uv3.train.fsdp2_trainer \
  --config "${CONFIG}" &
fi
train_pid=$!

if [[ "${UV3_MONITOR}" == "1" ]]; then
  "${MONITOR_PYTHON}" "${MONITOR_ROOT}/backend/collect_gpu_telemetry.py" \
    --output "${output_dir}/gpu_telemetry.node${NODE_RANK}.jsonl" \
    --interval "${UV3_GPU_TELEMETRY_INTERVAL:-2}" \
    --pid "${train_pid}" \
    --node-rank "${NODE_RANK}" \
    >"${output_dir}/gpu_telemetry.node${NODE_RANK}.log" 2>&1 &
  telemetry_pid=$!
fi

if wait "${train_pid}"; then
  train_status=0
else
  train_status=$?
fi
train_pid=""
if [[ -n "${telemetry_pid}" ]]; then
  wait "${telemetry_pid}" 2>/dev/null || true
  telemetry_pid=""
fi
if [[ "${train_status}" != "0" ]]; then
  echo "[error] torchrun failed with status ${train_status}" >&2
  exit "${train_status}"
fi

# torchrun returns only after every worker on every node has exited.  All four
# training nodes are therefore clear before node rank 0 reuses its eight GPUs
# for the final student evaluation.  Other nodes can leave the allocation.
if [[ "${RUN_EVAL_AFTER_TRAIN:-1}" != "1" ]]; then
  echo "[eval] RUN_EVAL_AFTER_TRAIN=${RUN_EVAL_AFTER_TRAIN:-0}; skipping"
  exit 0
fi
if [[ "${UV3_BENCH_NO_CKPT:-0}" == "1" ]]; then
  echo "[eval] UV3_BENCH_NO_CKPT=1; smoke run has no checkpoint, skipping"
  exit 0
fi
if [[ "${NODE_RANK}" != "0" ]]; then
  echo "[eval] node rank ${NODE_RANK} finished; final evaluation runs on node rank 0"
  exit 0
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "[error] training completed but final checkpoint is missing: ${checkpoint}" >&2
  exit 1
fi

echo "[eval] training complete; starting final student evaluation on node rank 0"
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${output_dir}" \
CHECKPOINT="${checkpoint}" \
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/eval_4b_test.yaml}" \
MASTER_PORT="${EVAL_MASTER_PORT:-29685}" \
"${REPO_ROOT}/scripts/run_eval_4b_8gpu.sh" \
  > >(tee -a "${output_dir}/stdout.log") 2>&1 &
eval_pid=$!

if [[ "${UV3_MONITOR}" == "1" ]]; then
  "${MONITOR_PYTHON}" "${MONITOR_ROOT}/backend/collect_gpu_telemetry.py" \
    --output "${output_dir}/gpu_telemetry.node0.jsonl" \
    --interval "${UV3_GPU_TELEMETRY_INTERVAL:-2}" \
    --pid "${eval_pid}" \
    --node-rank 0 \
    >>"${output_dir}/gpu_telemetry.node0.log" 2>&1 &
  telemetry_pid=$!
fi

if wait "${eval_pid}"; then
  eval_status=0
else
  eval_status=$?
fi
eval_pid=""
if [[ -n "${telemetry_pid}" ]]; then
  wait "${telemetry_pid}" 2>/dev/null || true
  telemetry_pid=""
fi
if [[ "${eval_status}" != "0" ]]; then
  echo "[error] final evaluation failed with status ${eval_status}" >&2
  exit "${eval_status}"
fi
echo "[eval] final student evaluation complete"
