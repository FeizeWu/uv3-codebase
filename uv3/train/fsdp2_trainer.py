"""FSDP2 multi-GPU trainer for real pretraining (not the cached overfit).

Per-batch VAE encode + Qwen3.5 text encode (frozen, optionally node-local FSDP2) + MMDiT FSDP2 + Muon +
grad-accum + 6-stage timing + ckpt/resume (model+optimizer+RNG+data_status).
Run: torchrun --nproc_per_node=N -m uv3.train.fsdp2_trainer --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import shutil
import time
from dataclasses import asdict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..config import load_config, ExperimentConfig
from ..data.parquet_dataset import ParquetImageDataset
from ..modeling.vae import Flux2VAE
from ..modeling.qwen3_5 import Qwen3_5TextEncoder
from ..modeling.mmdit import MMDiT
from ..modeling.self_flow import (
    attach_self_flow_feature_captures,
    build_self_flow_projector,
    self_flow_feature_loss,
    build_self_flow_latents_continuous,
    validate_self_flow_config,
)
from ..modeling.flow import interpolate, velocity_target
from ..data.noise_scheduler import sample_timesteps, timestep_bin_sums
from ..data.dynamic_bucket_scheduler import (
    DistributedBucketController,
    DynamicJointBucketScheduler,
    LocalBucketController,
    smooth_weighted_schedule,
)
from ..optim.build_optimizers import build_optimizers
from .fsdp2 import (
    apply_fsdp2,
    apply_frozen_text_encoder_fsdp2,
    make_mesh,
    make_node_local_mesh,
    save_ckpt,
    load_ckpt,
)
from ..utils.timer import Timer


def _compile_training_modules(vae, mmdit, qwen, teacher, cfg, rank):
    """Compile one static graph per configured text/resolution bucket."""
    if not cfg.train.compile:
        return
    text_buckets = tuple(int(x) for x in getattr(cfg.train, "text_length_buckets", ()))
    if bool(getattr(cfg.data, "bucket", False)):
        from ..data.bucket_sampler import normalize_bucket_names
        resolution_bucket_count = len(
            normalize_bucket_names(getattr(cfg.data, "aspect_buckets", ()))
        )
    else:
        resolution_bucket_count = 1
    pad_text_to_max = bool(getattr(cfg.train, "pad_text_to_max_length", True))
    recompile_limit = "default"
    static_graph_count = max(1, len(text_buckets)) * max(1, resolution_bucket_count)
    if static_graph_count > 1:
        # Qwen's hybrid linear/full-attention stack creates more than one guarded
        # frame per sequence length (including bool/long internal mask variants).
        # The default limit of 8 makes a five-bucket run silently fall back to
        # eager mode, so reserve enough cache entries for all static graphs.
        from torch import _dynamo
        _dynamo.config.recompile_limit = max(
            int(_dynamo.config.recompile_limit),
            16 * static_graph_count,
        )
        recompile_limit = _dynamo.config.recompile_limit
    mmdit_compile_mode = str(getattr(cfg.train, "compile_mode", "default"))
    configured_text_mode = getattr(cfg.train, "text_encoder_compile_mode", None)
    text_compile_mode = (
        mmdit_compile_mode if configured_text_mode is None else str(configured_text_mode)
    )
    compile_kwargs = {
        "dynamic": False,
        "mode": mmdit_compile_mode,
    }
    mmdit.compile(**compile_kwargs)
    if teacher is not None:
        teacher.compile(**compile_kwargs)
    if bool(getattr(cfg.train, "compile_text_encoder", True)):
        qwen.language_model.compile(dynamic=False, mode=text_compile_mode)
    compile_vae = bool(getattr(cfg.train, "compile_vae", False))
    if compile_vae:
        vae_compile_mode = str(getattr(cfg.train, "vae_compile_mode", "default"))
        if resolution_bucket_count > 1 and vae_compile_mode in {
            "max-autotune", "reduce-overhead",
        }:
            raise ValueError(
                f"vae_compile_mode={vae_compile_mode!r} enables CUDA Graphs, which "
                "is unsafe with multiple resolution buckets; use "
                "'max-autotune-no-cudagraphs' or disable compile_vae"
            )
        # AutoencoderKLFlux2.encode() is decorated and wraps its tensor result in
        # a posterior object. Compile the tensor-only core instead. Each configured
        # resolution remains a separate static graph; CUDA Graphs stay disabled.
        vae.vae._encode = torch.compile(
            vae.vae._encode,
            dynamic=False,
            mode=vae_compile_mode,
        )
    if rank == 0:
        print(f"[train] torch.compile ON: mmdit=True teacher={teacher is not None} "
              f"text_encoder={bool(getattr(cfg.train, 'compile_text_encoder', True))} "
              f"vae={compile_vae} "
              f"vae_mode={str(getattr(cfg.train, 'vae_compile_mode', 'default')) if compile_vae else 'off'} "
              f"mmdit_text_buckets={bool(text_buckets) and not pad_text_to_max} "
              f"resolution_buckets={resolution_bucket_count} "
              f"dynamic=False mmdit_mode={mmdit_compile_mode} "
              f"text_mode={text_compile_mode} "
              f"recompile_limit={recompile_limit}", flush=True)


def _attention_mask(mmdit, text_valid, n_img, block_size, device):
    """Build a per-bucket static [text, image] Flex or SDPA padding mask."""
    bs, n_txt = text_valid.shape
    image_valid = torch.ones(bs, n_img, device=device, dtype=torch.bool)
    valid = torch.cat([text_valid.to(device=device, dtype=torch.bool), image_valid], dim=1)
    if mmdit._flex:
        from .flex_attn import build_padding_block_mask
        heads = int(getattr(mmdit.transformer.config, "num_attention_heads", 1))
        return build_padding_block_mask(valid, heads, block_size=block_size, _compile=False)
    additive = torch.zeros(bs, 1, 1, n_txt + n_img, device=device, dtype=mmdit.dtype)
    additive.masked_fill_(~valid[:, None, None, :], torch.finfo(mmdit.dtype).min)
    return additive


def format_resolution_bucket_loss(reduced, aspect_buckets, sf_enabled, sf_coeff):
    """Convert globally reduced [count, FM sum, SF sum] tensors to JSON metrics."""
    counts, fm_sums, sf_sums = reduced
    safe_counts = counts.clamp_min(1.0)
    fm_loss = fm_sums / safe_counts
    sf_loss = sf_sums / safe_counts
    total_loss = fm_loss + (float(sf_coeff) * sf_loss if sf_enabled else 0.0)
    return {
        "names": [bucket.name for bucket in aspect_buckets],
        "width": [int(bucket.width) for bucket in aspect_buckets],
        "height": [int(bucket.height) for bucket in aspect_buckets],
        "image_tokens": [int(bucket.image_tokens) for bucket in aspect_buckets],
        "count": [int(value) for value in counts.tolist()],
        "fm_loss": [float(value) for value in fm_loss.tolist()],
        "self_flow_loss": (
            [float(value) for value in sf_loss.tolist()] if sf_enabled else None
        ),
        "total_loss": [float(value) for value in total_loss.tolist()],
    }


def _reshard_fsdp2_modules_(model: nn.Module) -> None:
    """Restore DTensor parameter views for every composable-FSDP unit."""
    for module in model.modules():
        reshard = getattr(module, "reshard", None)
        if callable(reshard):
            reshard()


@torch.no_grad()
def _ema_update_local_shards_(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update identically sharded FSDP2 parameters without full materialization."""
    _reshard_fsdp2_modules_(teacher)
    _reshard_fsdp2_modules_(student)
    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    if teacher_params.keys() != student_params.keys():
        missing = sorted(teacher_params.keys() - student_params.keys())
        extra = sorted(student_params.keys() - teacher_params.keys())
        raise RuntimeError(f"EMA parameter names mismatch: missing={missing[:3]} extra={extra[:3]}")
    weight = 1.0 - decay
    for name, teacher_param in teacher_params.items():
        student_param = student_params[name]
        teacher_local = getattr(teacher_param, "to_local", None)
        student_local = getattr(student_param, "to_local", None)
        if (teacher_local is None) != (student_local is None):
            raise RuntimeError(
                f"EMA parameter {name} has mismatched sharding: "
                f"teacher={type(teacher_param).__name__} student={type(student_param).__name__}"
            )
        dst = teacher_local() if teacher_local is not None else teacher_param
        src = student_local() if student_local is not None else student_param
        if dst.shape != src.shape:
            raise RuntimeError(
                f"EMA local shard shape mismatch at {name}: {dst.shape} != {src.shape}"
            )
        dst.lerp_(src, weight)


def _assert_fp32_training_storage(
    model: nn.Module,
    teacher: nn.Module | None,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    require_optimizer_state: bool,
) -> None:
    """Enforce FP32 master parameters, EMA storage, and Adam moments."""
    for owner, module in (("student", model), ("teacher", teacher)):
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                raise RuntimeError(
                    f"{owner} master parameter {name} must be FP32, got {parameter.dtype}"
                )
    moment_count = 0
    for optimizer_name, optimizer in optimizers.items():
        for state in optimizer.state.values():
            for state_name in ("exp_avg", "exp_avg_sq"):
                value = state.get(state_name)
                if value is None:
                    continue
                moment_count += 1
                if value.dtype != torch.float32:
                    raise RuntimeError(
                        f"optimizer {optimizer_name} {state_name} must be FP32, "
                        f"got {value.dtype}"
                    )
    if require_optimizer_state and not moment_count:
        raise RuntimeError("optimizer has no Adam moment state after an optimizer step")


def _cast_adam_moments_fp32_(optimizers: dict[str, torch.optim.Optimizer]) -> None:
    """Upgrade legacy BF16 Adam moments after loading an older checkpoint."""
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for state_name in ("exp_avg", "exp_avg_sq"):
                value = state.get(state_name)
                if value is not None and value.dtype != torch.float32:
                    state[state_name] = value.float()


def _reconcile_monitor_metrics_for_resume(out_dir: str, start_step: int) -> dict:
    """Keep one monotonic metrics branch ending immediately before the checkpoint."""
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    event = {
        "type": "resume",
        "timestamp": time.time(),
        "checkpoint_step": int(start_step),
        "kept_metric_rows": 0,
        "discarded_metric_rows": 0,
        "metrics_archive": None,
    }
    if os.path.exists(metrics_path):
        valid_before: dict[int, str] = {}
        discarded: list[str] = []
        with open(metrics_path, encoding="utf-8", errors="replace") as metrics_file:
            for line in metrics_file:
                try:
                    row = json.loads(line)
                    step = int(row["step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    discarded.append(line)
                    continue
                if step < start_step:
                    # A run produced by an older launcher may already contain
                    # duplicate steps. Keep the newest value for each step.
                    valid_before[step] = line if line.endswith("\n") else line + "\n"
                else:
                    discarded.append(line)
        with open(metrics_path, encoding="utf-8", errors="replace") as metrics_file:
            original_rows = sum(1 for _ in metrics_file)
        needs_rewrite = len(valid_before) + len(discarded) != original_rows or bool(discarded)
        if needs_rewrite:
            timestamp = int(event["timestamp"])
            archive_path = os.path.join(
                out_dir,
                f"metrics.pre_resume_step_{start_step:08d}_{timestamp}.jsonl",
            )
            shutil.copyfile(metrics_path, archive_path)
            temporary = metrics_path + ".resume.tmp"
            with open(temporary, "w", encoding="utf-8") as metrics_file:
                for step in sorted(valid_before):
                    metrics_file.write(valid_before[step])
            os.replace(temporary, metrics_path)
            event["metrics_archive"] = os.path.basename(archive_path)
        event["kept_metric_rows"] = len(valid_before)
        event["discarded_metric_rows"] = len(discarded)
    with open(os.path.join(out_dir, "events.jsonl"), "a", encoding="utf-8") as events_file:
        events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


class DataPrefetcher:
    """Async next batch -> device (overlap data load with compute)."""

    def __init__(self, loader, dev, dtype, resume_state: dict | None = None):
        self.loader = iter(loader)
        self.dev = dev
        self.dtype = dtype
        self.stream = torch.cuda.Stream()
        self._next = None
        self._next_cpu = None
        restored_batch = (resume_state or {}).get("next_batch")
        self._preload(restored_batch)

    def _preload(self, batch=None):
        if batch is None:
            try:
                batch = next(self.loader)
            except StopIteration:
                self._next = None
                self._next_cpu = None
                return
        self._next_cpu = batch
        with torch.cuda.stream(self.stream):
            pv = batch["pixel_values"].to(self.dev, non_blocking=True).to(self.dtype)
            self._next = {"pixel_values": pv, "text": batch["text"]}
            for key in ("resolution_bucket", "image_height", "image_width"):
                if key in batch:
                    self._next[key] = batch[key]
            for key in (
                "joint_token_length",
                "bucket_promoted_samples",
                "bucket_buffer_samples",
                "bucket_source_samples",
                "bucket_scheduled_samples",
                "bucket_emitted_samples",
                "bucket_decode_wait_seconds",
                "bucket_scheduler",
            ):
                if key in batch:
                    self._next[key] = batch[key]
            if "text_bucket_length" in batch:
                self._next["text_bucket_length"] = batch["text_bucket_length"]
            if "input_ids" in batch:
                # Keep pinned tokens on CPU here.  Copying the next text batch
                # on the image-prefetch stream overlaps small H2D operations
                # with large training kernels and regresses H20 throughput.
                # The ordered current-stream copy below is still asynchronous
                # from pinned memory and avoids retokenization.
                self._next["input_ids"] = batch["input_ids"]
                self._next["attention_mask"] = batch["attention_mask"]

    def state_dict(self) -> dict:
        def cpu_clone(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            return copy.deepcopy(value)

        if self._next_cpu is None:
            return {"next_batch": None}
        if "_resume_spec" in self._next_cpu:
            # Online tar batches can be reconstructed from compact descriptors;
            # do not gather decoded image tensors from every rank into rank 0.
            return {"next_spec": copy.deepcopy(self._next_cpu["_resume_spec"])}
        return {
            "next_batch": {
                key: cpu_clone(value) for key, value in self._next_cpu.items()
            }
        }

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        out = self._next
        self._preload()
        return out


class LengthBucketBatcher:
    """Form static text-length/resolution batches on an identical rank schedule."""

    def __init__(
        self,
        loader,
        tokenizer,
        batch_size,
        buckets=(),
        weights=(),
        resolution_buckets=(),
        resolution_weights=(),
        max_queue_batches=8,
    ):
        self.loader = loader
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.buckets = tuple(int(x) for x in buckets)
        self.weights = tuple(int(x) for x in weights)
        self.resolution_buckets = tuple(resolution_buckets)
        self.resolution_names = tuple(bucket.name for bucket in self.resolution_buckets)
        self.resolution_weights = tuple(int(x) for x in resolution_weights)
        self.max_queue_samples = max(1, int(max_queue_batches)) * self.batch_size
        self._slot = 0
        self._schedule_cursor = 0
        self._token_buffers = {}
        self.dropped_samples = 0
        if self.buckets and tuple(sorted(self.buckets)) != self.buckets:
            raise ValueError(f"text_length_buckets must be sorted: {self.buckets}")
        if self.buckets and (
            len(self.weights) != len(self.buckets) or any(x < 1 for x in self.weights)
        ):
            raise ValueError(
                "text_length_bucket_weights must contain one positive integer "
                f"per bucket: buckets={self.buckets}, weights={self.weights}"
            )
        if self.resolution_buckets and (
            len(self.resolution_weights) != len(self.resolution_buckets)
            or any(x < 1 for x in self.resolution_weights)
        ):
            raise ValueError(
                "aspect_bucket_weights must contain one positive integer per "
                f"bucket: buckets={self.resolution_names}, weights={self.resolution_weights}"
            )
        if not self.buckets and not self.resolution_buckets:
            raise ValueError("LengthBucketBatcher requires text or resolution buckets")

        text_schedule = tuple(zip(self.buckets, self.weights)) or ((None, 1),)
        resolution_schedule = (
            tuple(zip(self.resolution_names, self.resolution_weights)) or ((None, 1),)
        )
        self._schedule = [
            (resolution_name, text_length)
            for resolution_name, resolution_weight in resolution_schedule
            for text_length, text_weight in text_schedule
            for _ in range(resolution_weight * text_weight)
        ]
        random.Random(0).shuffle(self._schedule)

    def _tokenize_for_bucket(self, text):
        if not self.buckets:
            return None, None
        token_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.buckets[-1],
        )["input_ids"]
        length = len(token_ids)
        bucket = next((bucket for bucket in self.buckets if length <= bucket), self.buckets[-1])
        return bucket, torch.tensor(token_ids, dtype=torch.long)

    def _next_target(self):
        target = self._schedule[self._schedule_cursor % len(self._schedule)]
        self._schedule_cursor += 1
        return target

    def __iter__(self):
        queues = {target: [] for target in set(self._schedule)}
        source = iter(self.loader)
        while True:
            target = self._next_target()
            while len(queues[target]) < self.batch_size:
                try:
                    sample = next(source)
                except StopIteration:
                    return
                text_bucket, input_ids = self._tokenize_for_bucket(sample["text"])
                if input_ids is not None:
                    sample["input_ids"] = input_ids
                resolution_bucket = (
                    sample.get("resolution_bucket") if self.resolution_buckets else None
                )
                key = (resolution_bucket, text_bucket)
                if key not in queues:
                    raise RuntimeError(
                        f"dataset emitted unscheduled bucket {key}; "
                        f"known={sorted(queues, key=str)}"
                    )
                queue = queues[key]
                queue.append(sample)
                if len(queue) > self.max_queue_samples:
                    overflow = len(queue) - self.max_queue_samples
                    del queue[:overflow]
                    self.dropped_samples += overflow
            samples = queues[target][:self.batch_size]
            del queues[target][:self.batch_size]
            pin_memory = torch.cuda.is_available()
            slot = self._slot
            self._slot = 1 - self._slot
            # Keep image collation on the existing pageable path.  Pinning the
            # much larger image tensor made its H2D copy overlap the current
            # step and reduced H20 training throughput.  Token tensors are tiny
            # and benefit from persistent pinned buffers without that contention.
            pixel_values = torch.stack(
                [sample["pixel_values"] for sample in samples]
            )
            resolution_name, text_target = target
            batch = {
                "pixel_values": pixel_values,
                "text": [sample["text"] for sample in samples],
                "resolution_bucket": resolution_name or samples[0].get("resolution_bucket", "square"),
                "image_height": int(pixel_values.shape[-2]),
                "image_width": int(pixel_values.shape[-1]),
            }
            if text_target is not None:
                token_key = (slot, text_target)
                buffers = self._token_buffers.get(token_key)
                if buffers is None:
                    buffers = (
                        torch.empty(
                            (self.batch_size, text_target),
                            dtype=torch.long,
                            pin_memory=pin_memory,
                        ),
                        torch.empty(
                            (self.batch_size, text_target),
                            dtype=torch.long,
                            pin_memory=pin_memory,
                        ),
                    )
                    self._token_buffers[token_key] = buffers
                input_ids, attention_mask = buffers
                input_ids.fill_(int(self.tokenizer.pad_token_id))
                attention_mask.zero_()
                for index, sample in enumerate(samples):
                    ids = sample["input_ids"]
                    length = ids.numel()
                    input_ids[index, :length].copy_(ids)
                    attention_mask[index, :length] = 1
                batch.update({
                    "text_bucket_length": text_target,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                })
            yield batch


def _smooth_weighted_schedule(buckets, weights):
    """Return a low-discrepancy period with exactly the requested integer weights."""
    return smooth_weighted_schedule(buckets, weights)


class OnlineJointBucketBatcher:
    """Online tokenize-once scheduler: global text/joint bucket, local aspect batch.

    The source yields compact tar descriptors. Captions are tokenized in CPU
    batches exactly once. A short caption may be promoted to a larger static
    text bucket; within each rank all selected images share one aspect shape.
    Tar seek/decode begins only after selection and is pipelined one or more
    batches ahead of GPU consumption.
    """

    def __init__(
        self,
        loader,
        tokenizer,
        batch_size,
        text_buckets,
        text_weights,
        resolution_buckets,
        *,
        tokenize_batch_size=256,
        max_buffer_samples=8192,
        decode_workers=4,
        decode_prefetch_batches=2,
        decode_fn=None,
        dynamic_scheduler=False,
        bucket_controller=None,
        lookahead_per_slot=512,
        soft_buffer_limit=6144,
        long_term_window_per_rank=50_000,
        long_term_safety_margin=0.02,
    ):
        self.loader = loader
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.text_buckets = tuple(int(value) for value in text_buckets)
        self.text_weights = tuple(int(value) for value in text_weights)
        self.resolution_buckets = tuple(resolution_buckets)
        self.resolution_names = tuple(bucket.name for bucket in self.resolution_buckets)
        self.max_image_tokens = max(bucket.image_tokens for bucket in self.resolution_buckets)
        self.tokenize_batch_size = max(1, int(tokenize_batch_size))
        self.max_buffer_samples = max(self.batch_size, int(max_buffer_samples))
        self.decode_workers = max(1, int(decode_workers))
        self.decode_prefetch_batches = max(1, int(decode_prefetch_batches))
        self.decode_fn = decode_fn
        if tuple(sorted(self.text_buckets)) != self.text_buckets:
            raise ValueError(f"text buckets must be sorted: {self.text_buckets}")
        if not self.resolution_buckets:
            raise ValueError("online joint bucketing requires resolution buckets")
        self.dynamic_enabled = bool(dynamic_scheduler)
        self.lookahead_per_slot = int(lookahead_per_slot)
        self.soft_buffer_limit = int(soft_buffer_limit)
        self.scheduler = None
        self.bucket_controller = None
        if self.dynamic_enabled:
            self.scheduler = DynamicJointBucketScheduler(
                self.text_buckets,
                self.text_weights,
                batch_size=self.batch_size,
                resolution_count=len(self.resolution_names),
                long_term_window_per_rank=long_term_window_per_rank,
                safety_margin=long_term_safety_margin,
            )
            self.bucket_controller = bucket_controller or LocalBucketController()
            minimum = self.scheduler.minimum_ready_buffer
            if self.lookahead_per_slot < minimum:
                raise ValueError(
                    f"lookahead_per_slot={self.lookahead_per_slot} cannot guarantee a "
                    f"ready resolution; minimum={minimum}"
                )
            if not minimum <= self.soft_buffer_limit <= self.max_buffer_samples:
                raise ValueError(
                    f"buffer limits must satisfy minimum={minimum} <= soft="
                    f"{self.soft_buffer_limit} <= hard={self.max_buffer_samples}"
                )
        elif bucket_controller is not None:
            raise ValueError("bucket_controller requires dynamic_scheduler=true")
        self._schedule = (
            self.scheduler.schedule
            if self.scheduler is not None
            else _smooth_weighted_schedule(self.text_buckets, self.text_weights)
        )
        self._schedule_cursor = 0
        self._slot = 0
        self._token_buffers = {}
        self._queues = {
            (resolution_name, text_bucket): deque()
            for resolution_name in self.resolution_names
            for text_bucket in self.text_buckets
        }
        self.source_samples = 0
        self.scheduled_samples = 0
        self.emitted_samples = 0
        self.promoted_samples = 0
        self.peak_buffer_samples = 0
        self.decode_error_samples = 0
        self.decode_fallback_duplicate_samples = 0
        self.source_cursors: dict[int, dict] = {}
        self.last_source_worker_id: int | None = None
        self._active_pending = None
        self._restored_pending_specs = deque()
        minimum_pigeonhole_buffer = (
            self.scheduler.minimum_ready_buffer
            if self.scheduler is not None
            else len(self._queues) * (self.batch_size - 1) + 1
        )
        if self.max_buffer_samples < minimum_pigeonhole_buffer:
            raise ValueError(
                "online joint bucket buffer cannot guarantee that any exact-native "
                f"queue becomes ready: limit={self.max_buffer_samples} "
                f"minimum={minimum_pigeonhole_buffer}"
            )

    @property
    def buffered_samples(self):
        return sum(len(queue) for queue in self._queues.values())

    @staticmethod
    def _checkpoint_spec(spec):
        """Copy a pending spec without persisting nondeterministic timing data."""
        saved = copy.deepcopy(spec)
        if len(saved) == 5:
            saved[4]["control_wait_ms"] = 0.0
        return saved

    def _native_text_bucket(self, length):
        return next(
            (bucket for bucket in self.text_buckets if length <= bucket),
            self.text_buckets[-1],
        )

    def _tokenize_more(self, source, max_samples=None):
        available = self.max_buffer_samples - self.buffered_samples
        if available <= 0:
            counts = {
                f"{resolution}:{text}": len(queue)
                for (resolution, text), queue in self._queues.items()
                if queue
            }
            raise RuntimeError(
                "online joint bucket descriptor buffer is full with no batch "
                f"eligible for the scheduled target; limit={self.max_buffer_samples} "
                f"queues={counts}. Check text bucket weights against the active tokenizer."
            )
        samples = []
        # Never overshoot the hard limit merely because tokenization is batched.
        # The configured schedule remains authoritative; a distribution drift
        # still fails clearly rather than silently dropping descriptors.
        count = min(self.tokenize_batch_size, available)
        if max_samples is not None:
            count = min(count, max(0, int(max_samples)))
        for _ in range(count):
            try:
                samples.append(next(source))
            except StopIteration:
                break
        if not samples:
            return False
        encoded = self.tokenizer(
            [sample["text"] for sample in samples],
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_buckets[-1],
            padding=False,
        )["input_ids"]
        if len(encoded) != len(samples):
            raise RuntimeError(
                f"tokenizer returned {len(encoded)} rows for {len(samples)} captions"
            )
        for sample, token_ids in zip(samples, encoded):
            worker_id = sample.get("worker_id")
            if worker_id is not None:
                worker_id = int(worker_id)
                self.source_cursors[worker_id] = {
                    "epoch": int(sample["epoch"]),
                    "shard_pos": int(sample["shard_pos"]),
                    "row_group": int(sample["row_group"]),
                    "row_pos": int(sample["row_pos"]),
                }
                self.last_source_worker_id = worker_id
            resolution_name = sample.get("resolution_bucket")
            if resolution_name not in self.resolution_names:
                raise RuntimeError(
                    f"dataset emitted unknown resolution bucket {resolution_name!r}; "
                    f"expected {self.resolution_names}"
                )
            token_ids = list(token_ids)
            native_bucket = self._native_text_bucket(len(token_ids))
            if self.scheduler is not None:
                self.scheduler.observe_native_target(native_bucket)
            sample["input_ids"] = token_ids
            sample["native_text_bucket"] = native_bucket
            self._queues[(resolution_name, native_bucket)].append(sample)
        self.source_samples += len(samples)
        buffered = self.buffered_samples
        self.peak_buffer_samples = max(self.peak_buffer_samples, buffered)
        if buffered > self.max_buffer_samples:
            raise AssertionError(f"buffer limit violated: {buffered} > {self.max_buffer_samples}")
        return len(samples)

    def state_dict(self) -> dict:
        pending = self._active_pending
        if pending is None:
            pending_specs = list(self._restored_pending_specs)
        else:
            pending_specs = [entry[0] for entry in pending]
        return {
            "version": 3,
            "dynamic_enabled": self.dynamic_enabled,
            "schedule": (
                self.scheduler.schedule if self.scheduler is not None else self._schedule
            ),
            "schedule_cursor": (
                self.scheduler.schedule_cursor
                if self.scheduler is not None else self._schedule_cursor
            ),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "queues": {
                key: list(copy.deepcopy(queue)) for key, queue in self._queues.items()
            },
            "pending_specs": [self._checkpoint_spec(spec) for spec in pending_specs],
            "source_cursors": copy.deepcopy(self.source_cursors),
            "last_source_worker_id": self.last_source_worker_id,
            "source_samples": self.source_samples,
            "scheduled_samples": self.scheduled_samples,
            "emitted_samples": self.emitted_samples,
            "promoted_samples": self.promoted_samples,
            "peak_buffer_samples": self.peak_buffer_samples,
            "decode_error_samples": self.decode_error_samples,
            "decode_fallback_duplicate_samples": self.decode_fallback_duplicate_samples,
        }

    def load_state_dict(self, state: dict) -> None:
        version = int(state.get("version", 0))
        if version not in {1, 2, 3}:
            raise ValueError(f"unsupported online bucket resume version: {state.get('version')}")
        saved_dynamic = bool(state.get("dynamic_enabled", False)) if version >= 2 else False
        if saved_dynamic != self.dynamic_enabled:
            raise ValueError(
                "dynamic bucket scheduler setting changed since checkpoint: "
                f"checkpoint={saved_dynamic} current={self.dynamic_enabled}"
            )
        if self.scheduler is not None:
            scheduler_state = state.get("scheduler")
            if scheduler_state is None:
                raise ValueError("dynamic checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(scheduler_state)
            self._schedule = self.scheduler.schedule
            self._schedule_cursor = self.scheduler.schedule_cursor
        else:
            if tuple(state.get("schedule", ())) != self._schedule:
                raise ValueError("online bucket schedule changed since checkpoint")
            self._schedule_cursor = int(state["schedule_cursor"])
        saved_queues = state.get("queues", {})
        if set(saved_queues) != set(self._queues):
            raise ValueError("online bucket queue keys changed since checkpoint")
        self._queues = {
            key: deque(copy.deepcopy(saved_queues[key])) for key in self._queues
        }
        self._restored_pending_specs = deque(copy.deepcopy(state.get("pending_specs", [])))
        if version < 3:
            # Versions 1/2 counted a spec as emitted when selected. Preserve
            # that meaning for already checkpointed pending specs, then use
            # the precise materialization-time counter for new selections.
            for spec in self._restored_pending_specs:
                if len(spec) == 5:
                    spec[4]["emission_accounted"] = True
        self.source_cursors = {
            int(worker): dict(cursor)
            for worker, cursor in state.get("source_cursors", {}).items()
        }
        last_worker = state.get("last_source_worker_id")
        self.last_source_worker_id = None if last_worker is None else int(last_worker)
        for name in (
            "source_samples", "emitted_samples", "promoted_samples",
            "peak_buffer_samples", "decode_error_samples",
            "decode_fallback_duplicate_samples",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self.scheduled_samples = int(
            state.get("scheduled_samples", state.get("emitted_samples", 0))
        )

    def _eligible_count(self, resolution_name, target):
        return sum(
            len(self._queues[(resolution_name, native)])
            for native in self.text_buckets
            if native <= target
        )

    def _ready_resolutions(self, target):
        return [
            resolution_name
            for resolution_name in self.resolution_names
            if self._eligible_count(resolution_name, target) >= self.batch_size
        ]

    def _local_ready_mask(self):
        if self.scheduler is None:
            raise RuntimeError("ready masks require the dynamic scheduler")
        counts = [
            [len(self._queues[(resolution, native)]) for native in self.text_buckets]
            for resolution in self.resolution_names
        ]
        return self.scheduler.ready_mask(counts)

    def _take_eligible(self, resolution_name, target, count):
        selected = []
        for native in reversed(self.text_buckets):
            if native > target:
                continue
            queue = self._queues[(resolution_name, native)]
            take = min(count - len(selected), len(queue))
            for _ in range(take):
                selected.append(queue.popleft())
            if len(selected) == count:
                break
        promoted = sum(
            int(sample["native_text_bucket"]) < target for sample in selected
        )
        return selected, promoted

    def _replacement_samples(self, source, resolution_name, target, count):
        while self._eligible_count(resolution_name, target) < count:
            # A malformed image must not turn unrelated queued descriptors into
            # a buffer overflow. The decoder will duplicate a valid pair from
            # the current batch when no bounded replacement can be admitted.
            if self.buffered_samples >= self.max_buffer_samples:
                return [], 0
            if not self._tokenize_more(source):
                return [], 0
        return self._take_eligible(resolution_name, target, count)

    def _select_spec(self, source):
        if self.scheduler is not None:
            return self._select_dynamic_spec(source)
        target = self._schedule[self._schedule_cursor % len(self._schedule)]
        while True:
            ready = self._ready_resolutions(target)
            if ready:
                break
            if not self._tokenize_more(source):
                return None
        # Prefer the aspect with the most native-fit rows, then total eligible
        # rows. This limits padding while allowing shorter captions to backfill.
        resolution_name = max(
            ready,
            key=lambda name: (
                len(self._queues[(name, target)]),
                self._eligible_count(name, target),
            ),
        )
        selected, promoted = self._take_eligible(
            resolution_name, target, self.batch_size,
        )
        if len(selected) != self.batch_size:
            raise AssertionError("eligible count and selected batch disagree")
        self._schedule_cursor += 1
        self.scheduled_samples += self.batch_size
        self.promoted_samples += promoted
        return target, resolution_name, selected, promoted, {
            "desired_target": target,
            "selected_target": target,
            "fallback": False,
            "fallback_to_max": False,
            "fallback_extra_text_tokens": 0,
            "lookahead_used": 0,
            "lookahead_exhausted": False,
            "local_ready_mask": None,
            "common_ready_mask": None,
            "control_wait_ms": 0.0,
            "schedule_version": 0,
            "base_weights": list(self.text_weights),
            "schedule_update_event": None,
            "emission_accounted": False,
        }

    def _select_dynamic_spec(self, source):
        scheduler = self.scheduler
        desired = scheduler.desired_target
        desired_bit = 1 << self.text_buckets.index(desired)
        lookahead_budget = min(
            self.lookahead_per_slot,
            max(0, self.soft_buffer_limit - self.buffered_samples),
        )
        lookahead_used = 0
        source_done = False
        local_mask = self._local_ready_mask()
        while not (local_mask & desired_bit) and lookahead_used < lookahead_budget:
            admitted = self._tokenize_more(
                source, max_samples=lookahead_budget - lookahead_used,
            )
            if not admitted:
                source_done = True
                break
            lookahead_used += admitted
            local_mask = self._local_ready_mask()
        locally_limited = not bool(local_mask & desired_bit) and (
            source_done
            or lookahead_used >= lookahead_budget
            or self.buffered_samples >= self.soft_buffer_limit
        )
        control = self.bucket_controller.synchronize(
            local_mask,
            has_completed_window=scheduler.has_completed_window,
            lookahead_exhausted=locally_limited,
        )
        if control.common_ready_mask == 0:
            raise RuntimeError(
                "dynamic joint bucket found no common ready target after bounded "
                f"lookahead; local_mask={local_mask:#x} buffer={self.buffered_samples} "
                f"lookahead={lookahead_used}/{lookahead_budget} source_done={source_done}"
            )
        selection = scheduler.select(control.common_ready_mask)
        target = selection.selected_target
        ready = self._ready_resolutions(target)
        if not ready:
            raise AssertionError(
                f"common target {target} is not locally ready; mask={local_mask:#x}"
            )
        resolution_name = max(
            ready,
            key=lambda name: (
                len(self._queues[(name, target)]),
                self._eligible_count(name, target),
            ),
        )
        selected, promoted = self._take_eligible(
            resolution_name, target, self.batch_size,
        )
        if len(selected) != self.batch_size:
            raise AssertionError("eligible count and selected batch disagree")
        scheduler.record_selection(selection)
        self.scheduled_samples += self.batch_size
        self.promoted_samples += promoted
        update_event = None
        control_wait_ms = control.wait_ms
        if scheduler.at_schedule_boundary and control.all_have_completed_window:
            local_counts = scheduler.peek_completed_window()
            weights, update_event, update_wait_ms = self.bucket_controller.update_weights(
                local_counts,
                old_weights=scheduler.base_weights,
                schedule_version=scheduler.schedule_version,
                safety_margin=scheduler.safety_margin,
            )
            scheduler.pop_completed_window()
            scheduler.install_weights(weights, update_event["schedule_version"])
            control_wait_ms += update_wait_ms
        self._schedule = scheduler.schedule
        self._schedule_cursor = scheduler.schedule_cursor
        return target, resolution_name, selected, promoted, {
            "desired_target": selection.desired_target,
            "selected_target": selection.selected_target,
            "fallback": selection.fallback,
            "fallback_to_max": selection.fallback_to_max,
            "fallback_extra_text_tokens": (
                selection.extra_text_tokens_per_sample * self.batch_size
            ),
            "lookahead_used": lookahead_used,
            "lookahead_exhausted": control.any_lookahead_exhausted,
            "local_ready_mask": local_mask,
            "common_ready_mask": control.common_ready_mask,
            "control_wait_ms": control_wait_ms,
            "schedule_version": scheduler.schedule_version,
            "base_weights": list(scheduler.base_weights),
            "schedule_update_event": update_event,
            "emission_accounted": False,
            "queue_counts": {
                f"{resolution}:{native}": len(self._queues[(resolution, native)])
                for resolution in self.resolution_names
                for native in self.text_buckets
            },
        }

    def _decode_spec(self, spec, futures, source, executor, decoder):
        from ..data.tar_dataset import TarDecodeFailure

        wait_start = time.perf_counter()
        target, resolution_name, samples, _, scheduler_meta = self._unpack_spec(spec)
        pixels = [None] * len(samples)
        pending = list(zip(range(len(samples)), futures))
        while pending:
            failed = []
            for index, future in pending:
                result = future.result()
                if isinstance(result, TarDecodeFailure):
                    failed.append((index, result))
                elif isinstance(result, torch.Tensor):
                    pixels[index] = result
                else:
                    raise TypeError(
                        f"descriptor decoder returned unsupported type {type(result)!r}"
                    )
            if not failed:
                break

            first_error_index = self.decode_error_samples + 1
            self.decode_error_samples += len(failed)
            for error_index, (index, failure) in enumerate(
                failed, start=first_error_index,
            ):
                if error_index <= 20:
                    sample = samples[index]
                    print(
                        "[data] skipping malformed image "
                        f"tar={sample.get('image_tar', '<custom>')} "
                        f"offset={sample.get('offset', '?')} "
                        f"size={sample.get('size', '?')} reason={failure.reason}",
                        flush=True,
                    )
                elif error_index == 21:
                    print("[data] further malformed-image warnings suppressed", flush=True)

            replacements, _ = self._replacement_samples(
                source, resolution_name, target, len(failed),
            )
            if replacements:
                pending = []
                for (index, _), replacement in zip(failed, replacements):
                    samples[index] = replacement
                    pending.append((index, executor.submit(decoder, replacement)))
                continue

            # This is reachable only at finite-source exhaustion. Reuse a full
            # valid image-text pair so every distributed rank still emits the
            # same batch shape and iteration count; malformed data never enters
            # training and image/caption alignment remains intact.
            donors = [index for index, pixel in enumerate(pixels) if pixel is not None]
            if not donors:
                raise RuntimeError(
                    "all samples in a terminal online-bucket batch failed image decode; "
                    "no valid pair is available to preserve distributed batch alignment"
                )
            for failure_offset, (index, _) in enumerate(failed):
                donor = donors[failure_offset % len(donors)]
                samples[index] = copy.copy(samples[donor])
                pixels[index] = pixels[donor]
                self.decode_fallback_duplicate_samples += 1
            pending = []

        decode_wait = time.perf_counter() - wait_start
        promoted = sum(
            int(sample["native_text_bucket"]) < target for sample in samples
        )
        return (
            target, resolution_name, samples, promoted, scheduler_meta,
        ), pixels, decode_wait

    def _unpack_spec(self, spec):
        if len(spec) == 5:
            return spec
        if len(spec) == 4 and not self.dynamic_enabled:
            target, resolution_name, samples, promoted = spec
            return target, resolution_name, samples, promoted, {
                "desired_target": target,
                "selected_target": target,
                "fallback": False,
                "fallback_to_max": False,
                "fallback_extra_text_tokens": 0,
                "lookahead_used": 0,
                "lookahead_exhausted": False,
                "local_ready_mask": None,
                "common_ready_mask": None,
                "control_wait_ms": 0.0,
                "schedule_version": 0,
                "base_weights": list(self.text_weights),
                "schedule_update_event": None,
                # Four-field specs only exist in pre-v3 checkpoints, where
                # selection already incremented emitted_samples.
                "emission_accounted": True,
            }
        raise ValueError(f"unsupported pending spec shape: {len(spec)}")

    def _materialize(self, spec, pixels, decode_wait):
        target, resolution_name, samples, promoted, scheduler_meta = self._unpack_spec(spec)
        if not scheduler_meta.get("emission_accounted", False):
            self.emitted_samples += len(samples)
            scheduler_meta["emission_accounted"] = True
        pixel_values = torch.stack(pixels)
        pin_memory = torch.cuda.is_available()
        slot = self._slot
        self._slot = 1 - self._slot
        token_key = (slot, target)
        buffers = self._token_buffers.get(token_key)
        if buffers is None:
            buffers = (
                torch.empty(
                    (self.batch_size, target), dtype=torch.long, pin_memory=pin_memory,
                ),
                torch.empty(
                    (self.batch_size, target), dtype=torch.long, pin_memory=pin_memory,
                ),
            )
            self._token_buffers[token_key] = buffers
        input_ids, attention_mask = buffers
        input_ids.fill_(int(self.tokenizer.pad_token_id))
        attention_mask.zero_()
        for index, sample in enumerate(samples):
            ids = sample["input_ids"]
            length = len(ids)
            input_ids[index, :length].copy_(torch.tensor(ids, dtype=torch.long))
            attention_mask[index, :length] = 1
        return {
            "pixel_values": pixel_values,
            "text": [sample["text"] for sample in samples],
            "resolution_bucket": resolution_name,
            "image_height": int(pixel_values.shape[-2]),
            "image_width": int(pixel_values.shape[-1]),
            "text_bucket_length": target,
            "joint_token_length": target + self.max_image_tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bucket_promoted_samples": promoted,
            "bucket_buffer_samples": self.buffered_samples,
            "bucket_source_samples": self.source_samples,
            "bucket_scheduled_samples": self.scheduled_samples,
            "bucket_emitted_samples": self.emitted_samples,
            "bucket_decode_wait_seconds": decode_wait,
            "bucket_decode_error_samples_cumulative": self.decode_error_samples,
            "bucket_decode_fallback_duplicates_cumulative": (
                self.decode_fallback_duplicate_samples
            ),
            "bucket_scheduler": copy.deepcopy(scheduler_meta),
            "_resume_spec": spec,
        }

    def __iter__(self):
        from ..data.tar_dataset import TarDescriptorDecoder

        source = iter(self.loader)
        decoder = self.decode_fn or TarDescriptorDecoder()
        pending = deque()
        source_done = False
        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for spec in self._restored_pending_specs:
                futures = [executor.submit(decoder, sample) for sample in spec[2]]
                pending.append((spec, futures))
            self._restored_pending_specs.clear()
            self._active_pending = pending
            try:
                while True:
                    while len(pending) < self.decode_prefetch_batches and not source_done:
                        spec = self._select_spec(source)
                        if spec is None:
                            source_done = True
                            break
                        futures = [executor.submit(decoder, sample) for sample in spec[2]]
                        pending.append((spec, futures))
                    if not pending:
                        return
                    spec, futures = pending.popleft()
                    spec, pixels, decode_wait = self._decode_spec(
                        spec, futures, source, executor, decoder,
                    )
                    yield self._materialize(spec, pixels, decode_wait)
            finally:
                self._active_pending = None


def _align_text_to_joint_length(text, mask, image_tokens, image_token_budget):
    """Append fully masked alignment slots so text+image has one global length."""
    alignment = int(image_token_budget) - int(image_tokens)
    if alignment < 0:
        raise RuntimeError(
            f"image tokens {image_tokens} exceed joint budget {image_token_budget}"
        )
    if alignment:
        text = F.pad(text, (0, 0, 0, alignment))
        mask = F.pad(mask, (0, alignment), value=0)
    return text, mask, alignment


def build(cfg: ExperimentConfig, dev, dtype):
    vae = Flux2VAE.from_pretrained(cfg.model.vae.pretrained, dtype=dtype, force_upcast=False).to(dev).eval()
    qwen = Qwen3_5TextEncoder.from_pretrained(
        cfg.model.qwen_vl.pretrained, max_length=cfg.model.qwen_vl.max_length, dtype=dtype
    ).to(dev).eval()
    cfg.model.flex_attention = bool(cfg.model.flex_attention or cfg.train.flex_attention)
    # Keep trainable storage and optimizer master/moment state in FP32. FSDP2's
    # MixedPrecisionPolicy casts only the forward/reduce path to BF16.
    mmdit = MMDiT.build(cfg.model.transformer, cfg.model, text_encoder=qwen).to(
        dev, dtype=torch.float32,
    )
    return vae, qwen, mmdit


def _maybe_quantize_text_encoder_fp8(qwen, cfg, rank):
    """Quantize the frozen Qwen Linear weights before torch.compile."""
    if not bool(getattr(cfg.train, "text_encoder_fp8", False)):
        return
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )

    scope = str(getattr(cfg.train, "text_encoder_fp8_scope", "all"))
    if scope not in {"all", "mlp", "mlp_middle"}:
        raise ValueError(
            "text_encoder_fp8_scope must be all, mlp, or mlp_middle, "
            f"got {scope!r}"
        )
    filter_fn = None
    if scope == "mlp":
        filter_fn = lambda module, fqn: (
            isinstance(module, nn.Linear) and ".mlp." in fqn
        )
    elif scope == "mlp_middle":
        mlp_start = int(getattr(cfg.train, "text_encoder_fp8_mlp_start", 4))
        mlp_end = int(getattr(cfg.train, "text_encoder_fp8_mlp_end", 28))
        if not 0 <= mlp_start < mlp_end:
            raise ValueError(
                f"invalid Qwen FP8 MLP layer range [{mlp_start}, {mlp_end})"
            )

        def filter_fn(module, fqn):
            parts = fqn.split(".")
            if not isinstance(module, nn.Linear) or "mlp" not in parts:
                return False
            try:
                layers_pos = parts.index("layers")
                layer = int(parts[layers_pos + 1])
            except (ValueError, IndexError):
                return False
            return mlp_start <= layer < mlp_end
    quantize_(
        qwen.language_model,
        # The torchao default mutates process-global Inductor settings. That
        # invalidates the already autotuned MMDiT cache and needlessly recompiles
        # all five buckets; FP8 scaled-mm itself does not require those changes.
        Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False),
        filter_fn=filter_fn,
    )
    # Quantization replaces BF16 parameters in-place. Release their allocator
    # cache before the five static Qwen graphs are materialized.
    torch.cuda.empty_cache()
    if rank == 0:
        print(
            f"[train] Qwen FP8 ON: tensorwise dynamic activation + FP8 weight "
            f"scope={scope}",
            flush=True,
        )


def _maybe_convert_mmdit_fp8(mmdit, cfg, rank):
    """Convert only token-level block Linear layers to trainable FP8.

    Timestep and norm-modulation Linear layers operate on the raw batch axis.
    With BS=12 their backward GEMMs violate scaled-mm's multiple-of-16
    constraint, and they are too small to benefit from FP8 anyway.
    """
    if not bool(getattr(cfg.train, "mmdit_fp8", False)):
        return
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    def token_block_linear(module, fqn):
        qualified = f".{fqn}."
        return isinstance(module, nn.Linear) and (
            ".transformer_blocks." in qualified
            or ".single_transformer_blocks." in qualified
        )

    fp8_fsdp_all_gather = bool(
        getattr(cfg.train, "mmdit_fp8_fsdp_all_gather", False)
    )
    fp8_precompute_scale = bool(
        getattr(cfg.train, "mmdit_fp8_precompute_scale", True)
    )
    convert_to_float8_training(
        mmdit,
        module_filter_fn=token_block_linear,
        config=Float8LinearConfig(
            enable_fsdp_float8_all_gather=fp8_fsdp_all_gather,
        ),
    )
    if rank == 0:
        converted = sum(
            type(module).__name__ == "Float8Linear" for module in mmdit.modules()
        )
        print(
            f"[train] MMDiT FP8 ON: token-block Linear layers={converted} "
            f"fsdp_float8_all_gather={fp8_fsdp_all_gather} "
            f"precompute_scale={fp8_precompute_scale}",
            flush=True,
        )


def train(cfg: ExperimentConfig):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        rank = 0
    dev = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    vae, qwen, mmdit = build(cfg, dev, dtype)
    _maybe_quantize_text_encoder_fp8(qwen, cfg, rank)
    _maybe_convert_mmdit_fp8(mmdit, cfg, rank)
    fp8_fsdp_all_gather = bool(
        getattr(cfg.train, "mmdit_fp8", False)
        and getattr(cfg.train, "mmdit_fp8_fsdp_all_gather", False)
        and distributed
        and cfg.train.fsdp2
    )
    if fp8_fsdp_all_gather:
        # One packed scale all-reduce per optimizer step is substantially cheaper
        # than allowing every Float8Linear weight to reduce its scale separately.
        from torchao.float8.fsdp_utils import (
            precompute_float8_dynamic_scale_for_fsdp,
        )
    fp8_precompute_scale = fp8_fsdp_all_gather and bool(
        getattr(cfg.train, "mmdit_fp8_precompute_scale", True)
    )

    # Build the EMA teacher before FSDP so student and teacher can be sharded with
    # identical module boundaries and DTensor placements.  This lets EMA update
    # matching local shards instead of all-gathering every student parameter.
    sf_cfg = getattr(cfg.model, "self_flow", None)
    sf_enabled = bool(getattr(sf_cfg, "enabled", False)) if sf_cfg else False
    teacher = projector = student_cap = teacher_cap = None
    sf_student_has_text_prefix = False
    sf_teacher_has_text_prefix = False
    sf_coeff = 0.0
    sf_decay = 0.99
    sf_mask_ratio = 0.5
    sf_ratio = 0.5
    sf_timestep_mode = "ratio"
    if sf_enabled:
        validate_self_flow_config(sf_cfg)
        teacher = copy.deepcopy(mmdit).to(dev, dtype=torch.float32)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        projector = build_self_flow_projector(
            mmdit.inner_dim,
            projector_dim=getattr(sf_cfg, "projector_dim", None),
        ).to(dev, dtype=torch.float32)
        sf_coeff = float(getattr(sf_cfg, "coeff", 1.0))
        sf_decay = float(getattr(sf_cfg, "ema_decay", 0.9999))
        sf_mask_ratio = float(getattr(sf_cfg, "mask_ratio", 0.5))
        sf_ratio = float(getattr(sf_cfg, "ratio", 0.5))
        sf_timestep_mode = str(getattr(sf_cfg, "timestep_mode", "ratio"))
        (
            student_cap,
            teacher_cap,
            sf_student_has_text_prefix,
            sf_teacher_has_text_prefix,
        ) = (
            attach_self_flow_feature_captures(
                mmdit,
                teacher,
                student_depth=getattr(sf_cfg, "student_depth", None),
                teacher_depth=getattr(sf_cfg, "teacher_depth", None),
                student_depth_ratio=float(
                    getattr(sf_cfg, "student_depth_ratio", 0.3)
                ),
                teacher_depth_ratio=float(
                    getattr(sf_cfg, "teacher_depth_ratio", 0.7)
                ),
            )
        )
        if rank == 0:
            print(f"[train] self-flow enabled coeff={sf_coeff} ema_decay={sf_decay} "
                  f"mask_ratio={sf_mask_ratio} mode={sf_timestep_mode} "
                  f"student_depth={student_cap.global_depth} "
                  f"teacher_depth={teacher_cap.global_depth}", flush=True)

    # Model/projector initialization must match on every rank, while training
    # noise and timestep draws must not be identical across data-parallel ranks.
    torch.manual_seed(cfg.train.seed + rank)
    random.seed(cfg.train.seed + rank)
    np.random.seed(cfg.train.seed + rank)

    _compile_training_modules(vae, mmdit, qwen, teacher, cfg, rank)

    mesh = None
    if distributed and cfg.train.fsdp2:
        mesh = make_mesh(
            num_replicate=cfg.train.num_replicate,
            num_shard=cfg.train.num_shard,
        )
        if rank == 0:
            print(
                f"[train] MMDiT FSDP2 mesh shape={tuple(mesh.shape)} "
                f"dims={mesh.mesh_dim_names}",
                flush=True,
            )
        if bool(getattr(cfg.train, "fsdp_text_encoder", False)):
            qwen_mesh = make_node_local_mesh(
                shard_size=int(getattr(cfg.train, "text_encoder_shard_size", 8))
            )
            apply_frozen_text_encoder_fsdp2(qwen, qwen_mesh)
            if rank == 0:
                print(
                    f"[train] Qwen FSDP2 ON: node-local shard_size={qwen_mesh.size()} "
                    "reshard_after_forward=True",
                    flush=True,
                )
        apply_fsdp2(mmdit, mesh, reshard_after_forward=cfg.train.reshard_after_forward)
        if sf_enabled:
            # Teacher has no backward pass to trigger resharding.
            # Always reshard after forward so shard-local EMA sees matching DTensors.
            apply_fsdp2(teacher, mesh, reshard_after_forward=True)
            # shard the projector too so optimizer param groups are all-DTensor (avoid mixed foreach)
            from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
            mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            fully_shard(projector, mesh=mesh, mp_policy=mp, reshard_after_forward=cfg.train.reshard_after_forward)
    opt_model = nn.ModuleList([mmdit, projector]) if sf_enabled else mmdit
    optimizers, (n_m, n_a) = build_optimizers(opt_model, cfg)
    enforce_fp32_storage = not bool(getattr(cfg.train, "mmdit_fp8", False))
    if enforce_fp32_storage:
        _assert_fp32_training_storage(
            opt_model, teacher, optimizers, require_optimizer_state=False,
        )
    if rank == 0:
        npar = sum(p.numel() for p in mmdit.parameters() if p.requires_grad)
        print(f"[train] MMDiT params={npar:,} ({npar/1e9:.2f}B) muon={n_m} adam={n_a} world={os.environ.get('WORLD_SIZE','1')}", flush=True)

    out_dir = os.path.join(cfg.train.output_dir, cfg.train.run_name)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        config_tmp = os.path.join(out_dir, "config.yaml.tmp")
        with open(config_tmp, "w", encoding="utf-8") as config_file:
            yaml.safe_dump(asdict(cfg), config_file, sort_keys=False, allow_unicode=True)
        os.replace(config_tmp, os.path.join(out_dir, "config.yaml"))
        if cfg.train.monitor_enabled:
            run_tmp = os.path.join(out_dir, "run.json.tmp")
            with open(run_tmp, "w", encoding="utf-8") as run_file:
                json.dump(
                    {
                        "monitor_enabled": True,
                        "display_name": cfg.train.monitor_display_name or cfg.train.run_name,
                    },
                    run_file,
                    ensure_ascii=False,
                    indent=2,
                )
                run_file.write("\n")
            os.replace(run_tmp, os.path.join(out_dir, "run.json"))

    # RNG generator (defined BEFORE resume block so restore can reference it)
    g = torch.Generator(dev).manual_seed(123)

    # resume (load uses opt_model to match save's param-group keys; + RNG + data_status)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    resume_ckpt_path = os.environ.get("UV3_RESUME_CKPT", ckpt_path)
    if os.environ.get("UV3_RESUME_CKPT") and not os.path.exists(resume_ckpt_path):
        raise FileNotFoundError(f"explicit resume checkpoint is missing: {resume_ckpt_path}")
    start_step = 0
    resumed_epoch = 0
    storage_contract_checked = False
    restored_rng = None
    rank_data_status = None
    if os.path.exists(resume_ckpt_path):
        try:
            start_step, resume_payload = load_ckpt(
                opt_model, optimizers, resume_ckpt_path,
                extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
                return_payload=True,
                allow_self_flow_disable=not sf_enabled,
            )
            ck = resume_payload
            rank_rngs = ck.get("rng_by_rank")
            restored_rng = (
                rank_rngs[rank]
                if rank_rngs is not None and rank < len(rank_rngs)
                else ck.get("rng")
            )
            if restored_rng is not None:
                torch.set_rng_state(restored_rng.get("torch", restored_rng["py"]))
                if torch.cuda.is_available() and "cuda" in restored_rng:
                    torch.cuda.set_rng_state(restored_rng["cuda"])
                if "py_gen" in restored_rng:
                    g.set_state(restored_rng["py_gen"])
                if "python" in restored_rng:
                    random.setstate(restored_rng["python"])
                if "numpy" in restored_rng:
                    np.random.set_state(restored_rng["numpy"])
            data_statuses = ck.get("data_status_by_rank")
            rank_data_status = (
                data_statuses[rank]
                if data_statuses is not None and rank < len(data_statuses)
                else ck.get("data_status")
            ) or {}
            resumed_epoch = rank_data_status.get("epoch", 0) or 0
            if enforce_fp32_storage and start_step > 0:
                _cast_adam_moments_fp32_(optimizers)
                _assert_fp32_training_storage(
                    opt_model, teacher, optimizers, require_optimizer_state=True,
                )
                storage_contract_checked = True
            if rank == 0:
                resume_event = _reconcile_monitor_metrics_for_resume(out_dir, start_step)
                print(f"[train] resumed from step {start_step} (epoch {resumed_epoch})", flush=True)
                print(
                    "[train] monitor resume reconciled "
                    f"kept={resume_event['kept_metric_rows']} "
                    f"discarded={resume_event['discarded_metric_rows']} "
                    f"archive={resume_event['metrics_archive']}",
                    flush=True,
                )
                if rank_data_status.get("online_joint"):
                    print(
                        "[train] exact online data resume state found: per-worker "
                        "parquet cursors + bucket queues + decode/prefetch pending",
                        flush=True,
                    )
                else:
                    print(
                        "[train] WARNING: this checkpoint has no exact online data "
                        "pipeline state; non-production/legacy data order may restart",
                        flush=True,
                    )
        except Exception as e:
            if rank == 0:
                print(f"[train] resume failed ({type(e).__name__}: {e})", flush=True)
            raise RuntimeError(
                f"refusing to start fresh after checkpoint load failure: {resume_ckpt_path}"
            ) from e

    from ..data.bucket_sampler import build_aspect_buckets
    aspect_buckets = ()
    aspect_weights = ()
    online_joint_bucketing = False
    dynamic_bucket_enabled = bool(
        getattr(cfg.data, "dynamic_joint_bucket_scheduler", False)
    )
    text_buckets = tuple(
        int(x) for x in getattr(cfg.train, "text_length_buckets", ())
    )
    bucket_control_group = None
    bucket_controller = None
    if bool(getattr(cfg.data, "bucket", False)):
        aspect_buckets = build_aspect_buckets(
            cfg.data.image_size,
            getattr(cfg.data, "aspect_buckets", ()),
            getattr(cfg.data, "resolution_stride", 16),
        )
        if not aspect_buckets:
            raise ValueError("data.bucket=true requires at least one aspect bucket")
        online_joint_bucketing = bool(
            cfg.data.dataset == "tar"
            and getattr(cfg.data, "online_joint_bucketing", False)
        )
        if not online_joint_bucketing:
            configured_weights = tuple(
                int(x) for x in getattr(cfg.data, "aspect_bucket_weights", ())
            )
            aspect_weights = configured_weights or (1,) * len(aspect_buckets)
            if len(aspect_weights) != len(aspect_buckets) or any(x < 1 for x in aspect_weights):
                raise ValueError(
                    "data.aspect_bucket_weights must contain one positive integer "
                    f"per bucket: buckets={[bucket.name for bucket in aspect_buckets]} "
                    f"weights={aspect_weights}"
                )
        if rank == 0:
            print(
                "[train] resolution buckets="
                + ", ".join(
                    f"{bucket.name}:{bucket.width}x{bucket.height}"
                    f"/{bucket.image_tokens}tok"
                    for bucket in aspect_buckets
                ),
                flush=True,
            )
    if dynamic_bucket_enabled and not online_joint_bucketing:
        raise ValueError(
            "dynamic_joint_bucket_scheduler requires tar online_joint_bucketing"
        )
    if dynamic_bucket_enabled:
        if not text_buckets:
            raise ValueError(
                "dynamic_joint_bucket_scheduler requires non-empty text_length_buckets"
            )
        if distributed:
            bucket_control_group = torch.distributed.new_group(backend="gloo")
            bucket_controller = DistributedBucketController(
                bucket_control_group,
                bucket_count=len(cfg.train.text_length_buckets),
            )
        else:
            bucket_controller = LocalBucketController()

    pad_text_to_max = bool(getattr(cfg.train, "pad_text_to_max_length", True))
    text_bucket_weights = tuple(
        int(x) for x in getattr(cfg.train, "text_length_bucket_weights", ())
    ) if text_buckets else ()
    online_pipeline_signature = {
        "batch_size_per_gpu": int(cfg.train.batch_size_per_gpu),
        "text_buckets": list(text_buckets),
        "text_bucket_weights": list(text_bucket_weights),
        "tokenize_batch_size": int(getattr(cfg.data, "tokenize_batch_size", 256)),
        "bucket_buffer_max_samples": int(
            getattr(cfg.data, "bucket_buffer_max_samples", 8192)
        ),
        "dynamic_joint_bucket_scheduler": dynamic_bucket_enabled,
        "bucket_lookahead_per_slot": int(
            getattr(cfg.data, "bucket_lookahead_per_slot", 512)
        ),
        "bucket_soft_buffer_limit": int(
            getattr(cfg.data, "bucket_soft_buffer_limit", 6144)
        ),
        "bucket_long_term_window_per_rank": int(
            getattr(cfg.data, "bucket_long_term_window_per_rank", 50_000)
        ),
        "bucket_long_term_safety_margin": float(
            getattr(cfg.data, "bucket_long_term_safety_margin", 0.02)
        ),
        "tokenizer_pretrained": str(cfg.model.qwen_vl.pretrained),
        "tokenizer_max_length": int(cfg.model.qwen_vl.max_length),
    }

    # data: TarMetadataDataset (real data) or ParquetImageDataset (imagenet smoke)
    online_resume_state = None
    if cfg.data.dataset == "tar":
        from ..data.tar_dataset import TarMetadataDataset, validate_resume_signature
        ds = TarMetadataDataset(
            manifest_path=cfg.data.root,  # manifest path stored in root
            image_size=cfg.data.image_size,
            caption_field=getattr(cfg.data, "caption_field", "caption_qwen3_7_flash"),
            aspect_buckets=aspect_buckets,
            defer_image_decode=online_joint_bucketing,
        )
        ds.set_epoch(resumed_epoch)
        online_resume_state = (
            rank_data_status.get("online_joint") if rank_data_status else None
        )
        if start_step > 0 and online_joint_bucketing:
            if online_resume_state is None:
                raise RuntimeError(
                    "checkpoint predates exact online data resume state; refusing to "
                    "silently repeat the epoch prefix"
                )
            saved_signature = rank_data_status.get("data_signature")
            current_signature = ds.resume_signature(
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized() else 1,
                cfg.data.num_workers,
                pipeline=online_pipeline_signature,
            )
            validate_resume_signature(saved_signature, current_signature)
            batcher_resume = online_resume_state["batcher"]
            last_worker = batcher_resume.get("last_source_worker_id")
            worker_rotation = (
                0 if last_worker is None
                else (int(last_worker) + 1) % max(1, cfg.data.num_workers)
            )
            ds.set_resume(
                worker_cursors=batcher_resume.get("source_cursors", {}),
                worker_rotation=worker_rotation,
            )
    elif cfg.data.dataset == "fixed_tar":
        from ..data.fixed_tar_dataset import FixedTarSampleDataset
        # Small fixed sets are repeatedly buffered by text-length bucket. The
        # default file_descriptor sharing consumes one FD per queued tensor;
        # file_system keeps transient bucket buffering from exhausting RLIMIT.
        torch.multiprocessing.set_sharing_strategy("file_system")
        ds = FixedTarSampleDataset(
            cases_path=cfg.data.root,
            image_size=cfg.data.image_size,
            shuffle=True,
            aspect_buckets=aspect_buckets,
        )
        ds.set_epoch(resumed_epoch)
    else:
        ds = ParquetImageDataset(
            root=cfg.data.root, split=cfg.data.split, parquet_glob=cfg.data.parquet_glob,
            image_size=cfg.data.image_size, image_field=cfg.data.image_field,
            label_field=getattr(cfg.data, "label_field", "label"),
            aspect_buckets=aspect_buckets,
        )
        ds.set_epoch(resumed_epoch)
    data_loader_generator = torch.Generator().manual_seed(
        cfg.train.seed + rank + 10_000
    )
    if online_joint_bucketing:
        if not text_buckets:
            raise ValueError("online joint bucketing requires text_length_buckets")
        sample_loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=cfg.data.num_workers,
            generator=data_loader_generator,
        )
        loader = OnlineJointBucketBatcher(
            sample_loader,
            qwen.tokenizer,
            cfg.train.batch_size_per_gpu,
            text_buckets,
            text_bucket_weights,
            aspect_buckets,
            tokenize_batch_size=getattr(cfg.data, "tokenize_batch_size", 256),
            max_buffer_samples=getattr(cfg.data, "bucket_buffer_max_samples", 8192),
            decode_workers=getattr(cfg.data, "decode_workers", cfg.data.num_workers),
            decode_prefetch_batches=getattr(cfg.data, "decode_prefetch_batches", 2),
            dynamic_scheduler=dynamic_bucket_enabled,
            bucket_controller=bucket_controller,
            lookahead_per_slot=getattr(cfg.data, "bucket_lookahead_per_slot", 512),
            soft_buffer_limit=getattr(cfg.data, "bucket_soft_buffer_limit", 6144),
            long_term_window_per_rank=getattr(
                cfg.data, "bucket_long_term_window_per_rank", 50_000,
            ),
            long_term_safety_margin=getattr(
                cfg.data, "bucket_long_term_safety_margin", 0.02,
            ),
        )
        if online_resume_state is not None:
            loader.load_state_dict(online_resume_state["batcher"])
        if rank == 0:
            print(
                f"[train] online joint bucketing ON: text={text_buckets} "
                f"smooth_weights={text_bucket_weights} "
                f"joint_lengths={tuple(bucket + loader.max_image_tokens for bucket in text_buckets)} "
                "aspect=rank-local adaptive decode=after-selection "
                f"dynamic_scheduler={dynamic_bucket_enabled} "
                "buffer_drop=forbidden malformed=skip+backfill",
                flush=True,
            )
    elif text_buckets or aspect_buckets:
        sample_loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=cfg.data.num_workers,
            generator=data_loader_generator,
        )
        loader = LengthBucketBatcher(
            sample_loader,
            qwen.tokenizer,
            cfg.train.batch_size_per_gpu,
            text_buckets,
            text_bucket_weights,
            resolution_buckets=aspect_buckets,
            resolution_weights=aspect_weights,
        )
        if rank == 0:
            print(
                f"[train] static Qwen length buckets={text_buckets or 'off'} "
                f"weights={text_bucket_weights or 'off'} "
                f"mmdit_text_buckets={not pad_text_to_max}",
                flush=True,
            )
    else:
        loader = DataLoader(
            ds,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.data.num_workers,
            drop_last=True,
            generator=data_loader_generator,
        )
    prefetcher = DataPrefetcher(
        loader,
        dev,
        dtype,
        resume_state=(online_resume_state or {}).get("prefetcher"),
    )
    # DataLoader iterator creation consumes torch RNG for worker base seeds.
    # Restore checkpoint RNG again after rebuilding the exact data pipeline so
    # the next timestep/noise/Self-Flow mask draw matches uninterrupted training.
    if restored_rng is not None:
        torch.set_rng_state(restored_rng.get("torch", restored_rng["py"]))
        if torch.cuda.is_available() and "cuda" in restored_rng:
            torch.cuda.set_rng_state(restored_rng["cuda"])
        if "py_gen" in restored_rng:
            g.set_state(restored_rng["py_gen"])
        if "python" in restored_rng:
            random.setstate(restored_rng["python"])
        if "numpy" in restored_rng:
            np.random.set_state(restored_rng["numpy"])

    def capture_data_status(completed_step: int) -> dict:
        status = {"version": 1, "epoch": resumed_epoch, "step": int(completed_step)}
        if online_joint_bucketing:
            status["data_signature"] = ds.resume_signature(
                world_size, cfg.data.num_workers,
                pipeline=online_pipeline_signature,
            )
            batcher_state = loader.state_dict()
            prefetcher_state = prefetcher.state_dict()
            next_spec = prefetcher_state.pop("next_spec", None)
            if next_spec is not None:
                # The batcher has already yielded this spec to DataPrefetcher,
                # so put it back ahead of decode-pending specs for resume.
                batcher_state["pending_specs"].insert(
                    0, loader._checkpoint_spec(next_spec),
                )
            status["online_joint"] = {
                "batcher": batcher_state,
                "prefetcher": prefetcher_state,
            }
        return status
    timer = Timer()

    bs = cfg.train.batch_size_per_gpu
    accum = max(1, cfg.train.grad_accum)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    max_steps = cfg.train.max_steps
    log_every = cfg.train.log_every
    if rank == 0:
        # 计时口径写明:metrics.jsonl 的 timing 字段是 log_every 步窗口的累计秒数(非每步),
        # 因为 timer 在每次 log 后 reset。每步值 ≈ timing[k] / log_every。
        print(f"[train] 计时口径: metrics.jsonl timing 为 {log_every} 步窗口累计秒数(非每步);每步≈timing/{log_every}", flush=True)
    t0 = time.time()
    # 长时测速可通过环境变量丢弃初始化/缓存预热阶段，之后重新计时。
    # 只重置统计，不改变训练状态；用于整机独占的真实全管线吞吐测量。
    bench_warmup = int(os.environ.get("UV3_BENCH_WARMUP", "0"))
    # Diagnostic-only: scalarizing the returned DTensor norm introduces a GPU
    # synchronization, so keep this completely out of the production path.
    audit_grad_norm = os.environ.get("UV3_AUDIT_GRAD_NORM") == "1"
    grad_norm_samples: list[float] = []
    measured_steps = 0
    bucket_counts = {}
    resolution_bucket_counts = {}
    joint_bucket_counts = {}
    online_promoted_samples = 0
    online_decode_wait_seconds = 0.0
    online_buffer_peak = 0
    online_source_samples = 0
    online_emitted_samples = 0
    online_decode_error_samples = 0
    online_decode_fallback_duplicates = 0
    scheduler_telemetry_interval = int(
        getattr(cfg.data, "bucket_telemetry_interval_steps", 100)
    )
    scheduler_diagnostic_rate_limit = int(
        getattr(cfg.data, "bucket_diagnostic_rate_limit_steps", 1000)
    )
    if dynamic_bucket_enabled and scheduler_telemetry_interval < 1:
        raise ValueError("bucket_telemetry_interval_steps must be positive")
    if dynamic_bucket_enabled and scheduler_diagnostic_rate_limit < 1:
        raise ValueError("bucket_diagnostic_rate_limit_steps must be positive")
    scheduler_telemetry_path = os.path.join(out_dir, "bucket_scheduler.jsonl")
    scheduler_last_diagnostic_step = start_step - scheduler_diagnostic_rate_limit

    def new_scheduler_window(start: int) -> dict:
        bucket_count = len(text_buckets)
        return {
            "start_step": int(start),
            "steps": 0,
            "base_weights": list(text_bucket_weights),
            "schedule_version": 0,
            "desired": [0] * bucket_count,
            "selected": [0] * bucket_count,
            "fallback_matrix": [[0] * bucket_count for _ in range(bucket_count)],
            "fallback_steps": 0,
            "fallback_to_max_steps": 0,
            "fallback_extra_text_tokens_sum": 0,
            "buffer_sum": 0,
            "buffer_peak": 0,
            "lookahead_exhausted_steps": 0,
            "control_wait_ms_sum": 0.0,
            "control_wait_ms_max": 0.0,
            "hard_limit_hits": 0,
            "peak_snapshot": None,
        }

    scheduler_window = new_scheduler_window(start_step)

    def append_scheduler_record(record: dict) -> None:
        if rank != 0:
            return
        with open(scheduler_telemetry_path, "a", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")

    def observe_scheduler_batch(current_step: int, batch: dict) -> None:
        if not dynamic_bucket_enabled:
            return
        meta = batch.get("bucket_scheduler")
        if not isinstance(meta, dict):
            raise RuntimeError("dynamic bucket batch is missing scheduler metadata")
        desired = int(meta["desired_target"])
        selected = int(meta["selected_target"])
        desired_index = text_buckets.index(desired)
        selected_index = text_buckets.index(selected)
        scheduler_window["steps"] += 1
        scheduler_window["base_weights"] = list(meta["base_weights"])
        scheduler_window["schedule_version"] = int(meta["schedule_version"])
        scheduler_window["desired"][desired_index] += 1
        scheduler_window["selected"][selected_index] += 1
        if bool(meta["fallback"]):
            scheduler_window["fallback_steps"] += 1
            scheduler_window["fallback_matrix"][desired_index][selected_index] += 1
        if bool(meta["fallback_to_max"]):
            scheduler_window["fallback_to_max_steps"] += 1
        scheduler_window["fallback_extra_text_tokens_sum"] += int(
            meta["fallback_extra_text_tokens"]
        )
        buffer_samples = int(batch.get("bucket_buffer_samples", 0))
        scheduler_window["buffer_sum"] += buffer_samples
        scheduler_window["lookahead_exhausted_steps"] += int(
            bool(meta["lookahead_exhausted"])
        )
        wait_ms = float(meta["control_wait_ms"])
        scheduler_window["control_wait_ms_sum"] += wait_ms
        scheduler_window["control_wait_ms_max"] = max(
            scheduler_window["control_wait_ms_max"], wait_ms,
        )
        hard_limit = int(getattr(cfg.data, "bucket_buffer_max_samples", 8192))
        scheduler_window["hard_limit_hits"] += int(buffer_samples >= hard_limit)
        if buffer_samples >= scheduler_window["buffer_peak"]:
            scheduler_window["buffer_peak"] = buffer_samples
            scheduler_window["peak_snapshot"] = {
                "rank": rank,
                "buffer_samples": buffer_samples,
                "queue_counts": dict(meta.get("queue_counts", {})),
                "ready_mask": meta.get("local_ready_mask"),
                "common_ready_mask": meta.get("common_ready_mask"),
                "desired_target": desired,
                "selected_target": selected,
            }
        event = meta.get("schedule_update_event")
        if event is not None:
            append_scheduler_record({"step": int(current_step), **event})

    def flush_scheduler_window(current_step: int, *, force: bool = False) -> None:
        nonlocal scheduler_window, scheduler_last_diagnostic_step
        if not dynamic_bucket_enabled or scheduler_window["steps"] == 0:
            return
        if not force and scheduler_window["steps"] < scheduler_telemetry_interval:
            return
        local_sum = torch.tensor([
            float(scheduler_window["buffer_sum"]),
            float(scheduler_window["fallback_extra_text_tokens_sum"]),
            float(scheduler_window["control_wait_ms_sum"]),
            float(scheduler_window["hard_limit_hits"]),
        ], dtype=torch.float64, device="cpu")
        local_max = torch.tensor([
            float(scheduler_window["buffer_peak"]),
            float(scheduler_window["control_wait_ms_max"]),
        ], dtype=torch.float64, device="cpu")
        snapshots = [scheduler_window["peak_snapshot"]]
        if distributed:
            torch.distributed.all_reduce(
                local_sum, op=torch.distributed.ReduceOp.SUM,
                group=bucket_control_group,
            )
            torch.distributed.all_reduce(
                local_max, op=torch.distributed.ReduceOp.MAX,
                group=bucket_control_group,
            )
            snapshots = [None] * world_size
            torch.distributed.all_gather_object(
                snapshots,
                scheduler_window["peak_snapshot"],
                group=bucket_control_group,
            )
        if rank == 0:
            steps = int(scheduler_window["steps"])
            peak_snapshot = max(
                (snapshot for snapshot in snapshots if snapshot is not None),
                key=lambda snapshot: (snapshot["buffer_samples"], -snapshot["rank"]),
            )
            fallback_rate = scheduler_window["fallback_steps"] / steps
            fallback_max_rate = scheduler_window["fallback_to_max_steps"] / steps
            diagnostic_reasons = []
            if int(local_max[0].item()) > 4096:
                diagnostic_reasons.append("global_buffer_gt_4096")
            if fallback_rate > 0.05:
                diagnostic_reasons.append("fallback_rate_gt_5pct")
            if fallback_max_rate > 0.01:
                diagnostic_reasons.append("fallback_to_1024_rate_gt_1pct")
            if scheduler_window["lookahead_exhausted_steps"]:
                diagnostic_reasons.append("lookahead_exhausted")
            diagnostic = None
            if (
                diagnostic_reasons
                and current_step - scheduler_last_diagnostic_step
                >= scheduler_diagnostic_rate_limit
            ):
                diagnostic = {
                    "reasons": diagnostic_reasons,
                    "peak_rank": int(peak_snapshot["rank"]),
                    "queue_counts": peak_snapshot["queue_counts"],
                    "ready_mask": peak_snapshot["ready_mask"],
                    "common_ready_mask": peak_snapshot["common_ready_mask"],
                    "desired_target": peak_snapshot["desired_target"],
                    "selected_target": peak_snapshot["selected_target"],
                }
                scheduler_last_diagnostic_step = current_step
            append_scheduler_record({
                "event": "bucket_scheduler_window",
                "window_start_step": scheduler_window["start_step"],
                "window_end_step": int(current_step),
                "schedule_version": scheduler_window["schedule_version"],
                "base_weights": scheduler_window["base_weights"],
                "desired_target_counts": scheduler_window["desired"],
                "selected_target_counts": scheduler_window["selected"],
                "fallback_counts_by_pair": scheduler_window["fallback_matrix"],
                "fallback_steps": scheduler_window["fallback_steps"],
                "fallback_to_1024_steps": scheduler_window["fallback_to_max_steps"],
                "fallback_extra_text_tokens_sum": int(local_sum[1].item()),
                "global_buffer_max": int(local_max[0].item()),
                "global_buffer_mean": (
                    float(local_sum[0].item()) / (steps * world_size)
                ),
                "global_buffer_peak_rank": int(peak_snapshot["rank"]),
                "global_buffer_hard_limit_hits": int(local_sum[3].item()),
                "lookahead_exhausted_steps": scheduler_window[
                    "lookahead_exhausted_steps"
                ],
                "control_wait_ms_sum": float(local_sum[2].item()),
                "control_wait_ms_max": float(local_max[1].item()),
                "diagnostic": diagnostic,
            })
        scheduler_window = new_scheduler_window(current_step + 1)
    timestep_metrics = bool(getattr(cfg.train, "timestep_metrics", False))
    timestep_bin_counts = torch.zeros(10, device=dev, dtype=torch.float64)
    timestep_bin_fm_sums = torch.zeros_like(timestep_bin_counts)
    timestep_bin_sf_sums = torch.zeros_like(timestep_bin_counts)
    resolution_index = {
        bucket.name: index for index, bucket in enumerate(aspect_buckets)
    }
    resolution_loss_counts = torch.zeros(
        len(aspect_buckets), device=dev, dtype=torch.float64,
    )
    resolution_fm_sums = torch.zeros_like(resolution_loss_counts)
    resolution_sf_sums = torch.zeros_like(resolution_loss_counts)
    profile_start = int(os.environ.get("UV3_PROFILE_START", "-1"))
    profile_steps = int(os.environ.get("UV3_PROFILE_STEPS", "0"))
    profile_totals: dict[str, float] = {}
    profile_last = 0.0
    profile_active = False

    def profile_begin(current_step: int) -> None:
        nonlocal profile_last, profile_active
        profile_active = profile_steps > 0 and profile_start <= current_step < profile_start + profile_steps
        if profile_active:
            torch.cuda.synchronize()
            profile_last = time.perf_counter()

    def profile_mark(name: str) -> None:
        nonlocal profile_last
        if not profile_active:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        profile_totals[name] = profile_totals.get(name, 0.0) + now - profile_last
        profile_last = now
    for o in optimizers.values():
        o.zero_grad()

    loss = None
    step = start_step
    while step < max_steps:
        batch = prefetcher.next()
        if batch is None:
            ds.set_epoch(step)
            prefetcher = DataPrefetcher(loader, dev, dtype)
            batch = prefetcher.next()
        if online_joint_bucketing:
            online_promoted_samples += int(batch.get("bucket_promoted_samples", 0))
            online_decode_wait_seconds += float(batch.get("bucket_decode_wait_seconds", 0.0))
            online_buffer_peak = max(
                online_buffer_peak, int(batch.get("bucket_buffer_samples", 0))
            )
            online_source_samples = int(
                batch.get("bucket_source_samples", online_source_samples)
            )
            online_emitted_samples = int(
                batch.get("bucket_emitted_samples", online_emitted_samples)
            )
            online_decode_error_samples = int(batch.get(
                "bucket_decode_error_samples_cumulative", online_decode_error_samples,
            ))
            online_decode_fallback_duplicates = int(batch.get(
                "bucket_decode_fallback_duplicates_cumulative",
                online_decode_fallback_duplicates,
            ))
            observe_scheduler_batch(step, batch)
            flush_scheduler_window(step)
        timer.start()
        profile_begin(step)
        with torch.no_grad():
            latents = vae.encode_images(batch["pixel_values"])      # vae
        profile_mark("vae")
        timer.mark("vae")
        text_attn_mask = None
        with torch.no_grad():
            bucket_length = int(batch.get("text_bucket_length", cfg.model.qwen_vl.max_length))
            bucket_counts[bucket_length] = bucket_counts.get(bucket_length, 0) + 1
            if "input_ids" in batch:
                ids = batch["input_ids"].to(dev, non_blocking=True)
                mask = batch["attention_mask"].to(dev, non_blocking=True)
            else:
                ids, mask = qwen.tokenize(batch["text"], dev, max_length=bucket_length)
            text = qwen.encode_text(ids, mask)                  # qwen
            if pad_text_to_max and bucket_length < cfg.model.qwen_vl.max_length:
                pad_length = cfg.model.qwen_vl.max_length - bucket_length
                text = F.pad(text, (0, 0, 0, pad_length))
                mask = F.pad(mask, (0, pad_length), value=0)
            n_img = (latents.shape[-1] // 2) * (latents.shape[-2] // 2)
            alignment_tokens = 0
            if online_joint_bucketing:
                joint_image_budget = max(bucket.image_tokens for bucket in aspect_buckets)
                text, mask, alignment_tokens = _align_text_to_joint_length(
                    text, mask, n_img, joint_image_budget,
                )
            image_height, image_width = batch["pixel_values"].shape[-2:]
            token_stride = vae.scale_factor(max(image_height, image_width)) * cfg.model.patch_size
            if image_height % token_stride or image_width % token_stride:
                raise RuntimeError(
                    f"input {image_width}x{image_height} is not divisible by "
                    f"VAE+patch stride {token_stride}"
                )
            expected_img = (image_height // token_stride) * (image_width // token_stride)
            expected_text_base = (
                cfg.model.qwen_vl.max_length if pad_text_to_max else bucket_length
            )
            expected_text = expected_text_base + alignment_tokens
            if text.shape[1] != expected_text:
                raise RuntimeError(
                    f"static compile expects {expected_text} text tokens, "
                    f"got {text.shape[1]}"
                )
            if n_img != expected_img:
                raise RuntimeError(
                    f"static compile expects {expected_img} image tokens from "
                    f"{image_width}x{image_height}, got {n_img}"
                )
            if online_joint_bucketing:
                expected_joint = expected_text_base + joint_image_budget
                actual_joint = text.shape[1] + n_img
                configured_joint = int(batch.get("joint_token_length", expected_joint))
                if actual_joint != expected_joint or (
                    not pad_text_to_max and configured_joint != expected_joint
                ):
                    raise RuntimeError(
                        "joint-length alignment failed: "
                        f"text={text.shape[1]} image={n_img} actual={actual_joint} "
                        f"expected={expected_joint} configured={configured_joint}"
                    )
                joint_bucket_counts[expected_joint] = (
                    joint_bucket_counts.get(expected_joint, 0) + 1
                )
            resolution_name = str(
                batch.get("resolution_bucket", f"{image_width}x{image_height}")
            )
            resolution_bucket_counts[resolution_name] = (
                resolution_bucket_counts.get(resolution_name, 0) + 1
            )
            text_attn_mask = _attention_mask(
                mmdit, mask, n_img, cfg.train.block_size, dev,
            )
        profile_mark("qwen_and_mask")
        timer.mark("qwen")
        t = sample_timesteps(
            bs,
            dev,
            strategy=cfg.train.timestep_strategy,
            image_seq_len=n_img,
            shift=cfg.train.timestep_shift,
            logit_mean=cfg.train.timestep_logit_mean,
            logit_std=cfg.train.timestep_logit_std,
            base_seq_len=cfg.train.timestep_base_seq_len,
            max_seq_len=cfg.train.timestep_max_seq_len,
            base_shift=cfg.train.timestep_base_shift,
            max_shift=cfg.train.timestep_max_shift,
        )
        noise = torch.randn_like(latents)
        target_v = velocity_target(latents, noise).float()
        sf_loss_per_sample = None
        feat_loss = None
        if sf_enabled:
            paired_t = None
            if sf_timestep_mode == "independent":
                paired_t = sample_timesteps(
                    bs,
                    dev,
                    strategy=cfg.train.timestep_strategy,
                    image_seq_len=n_img,
                    shift=cfg.train.timestep_shift,
                    logit_mean=cfg.train.timestep_logit_mean,
                    logit_std=cfg.train.timestep_logit_std,
                    base_seq_len=cfg.train.timestep_base_seq_len,
                    max_seq_len=cfg.train.timestep_max_seq_len,
                    base_shift=cfg.train.timestep_base_shift,
                    max_shift=cfg.train.timestep_max_shift,
                )
            # per-token self-flow: mixed student latents + per-token timesteps
            student_latents, teach_lat, t_teach, token_t = build_self_flow_latents_continuous(
                latents, noise, t,
                mask_ratio=sf_mask_ratio, ratio=sf_ratio, timestep_mode=sf_timestep_mode,
                paired_t=paired_t, patch_size=cfg.model.patch_size,
            )
            # Student capture is an image-only double stream, or a full
            # [text, image] sequence when the architecture is pure-single.
            pred_v = mmdit(student_latents, text, t,
                           token_timesteps=token_t, text_attn_mask=text_attn_mask)
            fm_loss_per_sample = (pred_v.float() - target_v).square().flatten(1).mean(1)
            fm_loss = fm_loss_per_sample.mean()
            profile_mark("student_forward_and_fm_loss")
            with torch.no_grad():
                _ = teacher(teach_lat, text, t_teach, text_attn_mask=text_attn_mask)
            profile_mark("teacher_forward")
            s_feat = student_cap.features
            t_feat = teacher_cap.features
            n_txt_actual = text.shape[1]
            s_feat_img = s_feat[:, n_txt_actual:] if sf_student_has_text_prefix else s_feat
            t_feat_img = t_feat[:, n_txt_actual:] if sf_teacher_has_text_prefix else t_feat
            n_tok = min(s_feat_img.shape[1], t_feat_img.shape[1])
            proj_s = projector(s_feat_img[:, :n_tok])
            teacher_norm = F.normalize(t_feat_img[:, :n_tok].float().detach(), dim=-1)
            student_norm = F.normalize(proj_s.float(), dim=-1)
            sf_loss_per_sample = 1.0 - (teacher_norm * student_norm).sum(dim=-1).mean(dim=-1)
            feat_loss = sf_loss_per_sample.mean()
            loss = fm_loss + sf_coeff * feat_loss
            profile_mark("feature_loss")
        else:
            noisy = interpolate(latents, noise, t)
            pred_v = mmdit(noisy, text, t, text_attn_mask=text_attn_mask)
            fm_loss_per_sample = (pred_v.float() - target_v).square().flatten(1).mean(1)
            fm_loss = fm_loss_per_sample.mean()
            loss = fm_loss
            profile_mark("student_forward_and_fm_loss")
        if resolution_index:
            resolution_idx = resolution_index[resolution_name]
            resolution_loss_counts[resolution_idx] += fm_loss_per_sample.numel()
            resolution_fm_sums[resolution_idx] += fm_loss_per_sample.detach().double().sum()
            if sf_loss_per_sample is not None:
                resolution_sf_sums[resolution_idx] += sf_loss_per_sample.detach().double().sum()
        if timestep_metrics:
            bin_counts, bin_fm, bin_sf = timestep_bin_sums(
                t, fm_loss_per_sample, sf_loss_per_sample,
            )
            timestep_bin_counts += bin_counts
            timestep_bin_fm_sums += bin_fm
            timestep_bin_sf_sums += bin_sf
        timer.mark("forward")
        if not bool(torch.isfinite(loss.detach()).item()):
            resume_spec = batch.get("_resume_spec")
            source_samples = []
            if isinstance(resume_spec, (tuple, list)) and len(resume_spec) >= 3:
                for sample in resume_spec[2]:
                    source_samples.append(
                        {
                            "tar": os.path.basename(str(sample.get("image_tar", ""))),
                            "offset": sample.get("offset"),
                            "size": sample.get("size"),
                        }
                    )
            finite_stages = {
                "latents": bool(torch.isfinite(latents).all().item()),
                "text": bool(torch.isfinite(text).all().item()),
                "target": bool(torch.isfinite(target_v).all().item()),
                "prediction": bool(torch.isfinite(pred_v).all().item()),
            }
            if sf_enabled:
                finite_stages.update(
                    {
                        "student_features": bool(torch.isfinite(s_feat_img).all().item()),
                        "teacher_features": bool(torch.isfinite(t_feat_img).all().item()),
                        "projected_features": bool(torch.isfinite(proj_s).all().item()),
                    }
                )
            raise FloatingPointError(
                f"non-finite training loss before backward at step={step} rank={rank}: "
                f"fm={float(fm_loss.detach().item())} "
                f"self_flow={float(feat_loss.detach().item()) if feat_loss is not None else 'disabled'} "
                f"finite={finite_stages} sources={source_samples}"
            )
        (loss / accum).backward()                                     # backward
        profile_mark("backward")
        timer.mark("backward")
        if (step + 1) % accum == 0:
            clip_warmup = max(0, cfg.train.grad_clip_warmup_steps)
            clip_interval = max(0, cfg.train.grad_clip_interval)
            after_warmup = step - clip_warmup
            should_clip = cfg.train.grad_clip > 0 and (
                step < clip_warmup
                or (clip_interval > 0 and after_warmup % clip_interval == 0)
            )
            if should_clip:
                # Clip mmdit (FSDP2 DTensor) and projector (regular) SEPARATELY
                # because mixing DTensor and regular tensors breaks foreach.
                mmdit_grad_norm = torch.nn.utils.clip_grad_norm_(
                    mmdit.parameters(), cfg.train.grad_clip,
                    error_if_nonfinite=True,
                )
                if audit_grad_norm:
                    # Execute on every rank: DTensor scalar materialization may
                    # participate in collectives even though only rank 0 records it.
                    grad_norm_value = float(mmdit_grad_norm.detach().item())
                    if rank == 0:
                        grad_norm_samples.append(grad_norm_value)
                if projector is not None:
                    torch.nn.utils.clip_grad_norm_(
                        projector.parameters(), cfg.train.grad_clip,
                        error_if_nonfinite=True,
                    )
            profile_mark("grad_clip")
            for o in optimizers.values():
                o.step()                                             # optimizer
            if enforce_fp32_storage and not storage_contract_checked:
                _assert_fp32_training_storage(
                    opt_model, teacher, optimizers, require_optimizer_state=True,
                )
                storage_contract_checked = True
            profile_mark("optimizer")
            if fp8_precompute_scale:
                precompute_float8_dynamic_scale_for_fsdp(mmdit)
            for o in optimizers.values():
                o.zero_grad()
            profile_mark("zero_grad")
            # EMA matching local shards. Student and teacher use the same FSDP2
            # wrapping, so this is mathematically identical to full-parameter EMA
            # without a per-parameter all-gather or a replicated teacher.
            if sf_enabled:
                _ema_update_local_shards_(teacher, mmdit, sf_decay)
            profile_mark("ema_reshard_and_lerp")
        timer.mark("step")
        if profile_steps > 0 and step + 1 == profile_start + profile_steps and rank == 0:
            import json as _json
            per_step = {key: value / profile_steps for key, value in profile_totals.items()}
            print("[profile] " + _json.dumps({"steps": profile_steps, "seconds_per_step": per_step}, sort_keys=True), flush=True)
        if bench_warmup and step + 1 == start_step + bench_warmup:
            torch.cuda.synchronize()
            t0 = time.time()
            measured_steps = 0
            timer.reset()
            bucket_counts = {}
            resolution_bucket_counts = {}
            joint_bucket_counts = {}
            resolution_loss_counts.zero_()
            resolution_fm_sums.zero_()
            resolution_sf_sums.zero_()
            online_promoted_samples = 0
            online_decode_wait_seconds = 0.0
            online_buffer_peak = 0
            if rank == 0:
                print(f"[bench] warmup complete: {bench_warmup} steps; measurement starts now", flush=True)
        elif step + 1 > start_step + bench_warmup:
            measured_steps += 1
        # 下方 ts 是自上次 log 起 log_every 步的累计秒数(timer.reset 在块尾),非每步值。
        reduced_timestep_bins = None
        if timestep_metrics and step % log_every == 0:
            reduced = torch.stack(
                [timestep_bin_counts, timestep_bin_fm_sums, timestep_bin_sf_sums]
            )
            if distributed:
                torch.distributed.all_reduce(reduced)
            reduced_timestep_bins = reduced.cpu()
        reduced_resolution_loss = None
        if aspect_buckets and step % log_every == 0:
            reduced = torch.stack(
                [resolution_loss_counts, resolution_fm_sums, resolution_sf_sums]
            )
            if distributed:
                torch.distributed.all_reduce(reduced)
            reduced_resolution_loss = reduced.cpu()
        reduced_decode_counts = None
        if online_joint_bucketing and step % log_every == 0:
            reduced_decode_counts = torch.tensor(
                [online_decode_error_samples, online_decode_fallback_duplicates],
                device=dev,
                dtype=torch.int64,
            )
            if distributed:
                torch.distributed.all_reduce(reduced_decode_counts)
            reduced_decode_counts = reduced_decode_counts.cpu()
        if rank == 0 and step % log_every == 0:
            spd = measured_steps / (time.time() - t0 + 1e-9) if measured_steps else 0.0
            ts = timer.summary()
            mem_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            mem_pct = 100.0 * torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(dev).total_memory
            timing_str = " ".join(f"{k}={v:.3f}s" for k, v in ts.items()) if ts else ""
            bucket_str = (
                f" text_buckets={dict(sorted(bucket_counts.items()))}" if text_buckets else ""
            )
            if aspect_buckets:
                bucket_str += f" resolution_buckets={dict(sorted(resolution_bucket_counts.items()))}"
            if online_joint_bucketing:
                bucket_str += f" joint_buckets={dict(sorted(joint_bucket_counts.items()))}"
            grad_norm_stats = None
            grad_norm_str = ""
            timestep_stats = None
            resolution_loss_stats = None
            global_fm_loss = None
            global_sf_loss = None
            if reduced_timestep_bins is not None:
                counts, fm_sums, sf_sums = reduced_timestep_bins
                safe_counts = counts.clamp_min(1.0)
                timestep_stats = {
                    "edges": [i / 10 for i in range(11)],
                    "count": counts.to(torch.int64).tolist(),
                    "fm_loss": (fm_sums / safe_counts).tolist(),
                    "self_flow_loss": (sf_sums / safe_counts).tolist() if sf_enabled else None,
                }
            if reduced_resolution_loss is not None:
                resolution_loss_stats = format_resolution_bucket_loss(
                    reduced_resolution_loss, aspect_buckets, sf_enabled, sf_coeff,
                )
                total_count = reduced_resolution_loss[0].sum().clamp_min(1.0)
                global_fm_loss = float(reduced_resolution_loss[1].sum() / total_count)
                if sf_enabled:
                    global_sf_loss = float(reduced_resolution_loss[2].sum() / total_count)
            if audit_grad_norm and grad_norm_samples:
                grad_norm_stats = {
                    "min": min(grad_norm_samples),
                    "mean": sum(grad_norm_samples) / len(grad_norm_samples),
                    "max": max(grad_norm_samples),
                    "clip_count": sum(x > cfg.train.grad_clip for x in grad_norm_samples),
                    "count": len(grad_norm_samples),
                }
                grad_norm_str = (
                    f" grad_norm={grad_norm_stats['mean']:.4f}"
                    f"/{grad_norm_stats['max']:.4f}"
                    f" clipped={grad_norm_stats['clip_count']}/{grad_norm_stats['count']}"
                )
            online_bucket_stats = None
            online_bucket_str = ""
            if online_joint_bucketing:
                global_decode_errors = int(reduced_decode_counts[0])
                global_fallback_duplicates = int(reduced_decode_counts[1])
                online_bucket_stats = {
                    "promoted_samples": online_promoted_samples,
                    "buffer_peak_samples": online_buffer_peak,
                    "source_samples_cumulative": online_source_samples,
                    "emitted_samples_cumulative": online_emitted_samples,
                    "dropped_samples_cumulative": global_decode_errors,
                    "decode_fallback_duplicates_cumulative": (
                        global_fallback_duplicates
                    ),
                    "decode_wait_seconds": online_decode_wait_seconds,
                }
                online_bucket_str = (
                    f" online_bucket(promote={online_promoted_samples}"
                    f",buffer={online_buffer_peak}"
                    f",decode_wait={online_decode_wait_seconds:.3f}s"
                    f",drop={global_decode_errors}"
                    f",fallback_dup={global_fallback_duplicates})"
                )
            print(f"[train] step {step:5d} loss={loss.item():.4f} {spd:.2f}it/s mem={mem_gib:.1f}GiB({mem_pct:.1f}%){bucket_str}{online_bucket_str}{grad_norm_str} [{timing_str}]", flush=True)
            # JSONL metrics: timing_unit 自文档,说明 timing 是窗口和而非每步
            import json as _json
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as _jf:
                _json.dump({"step": step, "loss": loss.item(), "spd": spd,
                            "world_size": world_size,
                            "global_batch_size": bs * world_size * accum,
                            "max_memory_reserved_gib": mem_gib, "max_memory_reserved_pct": mem_pct,
                            "timing": ts, "timing_unit": f"sum_over_{log_every}_steps",
                            "text_bucket_counts": dict(sorted(bucket_counts.items())),
                            "resolution_bucket_counts": dict(sorted(resolution_bucket_counts.items())),
                            "joint_bucket_counts": dict(sorted(joint_bucket_counts.items())),
                            "online_bucket": online_bucket_stats,
                            "grad_norm": grad_norm_stats,
                            "fm_loss": global_fm_loss,
                            "self_flow_loss": global_sf_loss,
                            "resolution_bucket_loss": resolution_loss_stats,
                            "timestep_bins": timestep_stats}, _jf)
                _jf.write("\n")
            timer.reset()
            bucket_counts = {}
            resolution_bucket_counts = {}
            joint_bucket_counts = {}
            online_promoted_samples = 0
            online_decode_wait_seconds = 0.0
            online_buffer_peak = 0
            grad_norm_samples = []
        if timestep_metrics and step % log_every == 0:
            timestep_bin_counts.zero_()
            timestep_bin_fm_sums.zero_()
            timestep_bin_sf_sums.zero_()
        if aspect_buckets and step % log_every == 0:
            resolution_loss_counts.zero_()
            resolution_fm_sums.zero_()
            resolution_sf_sums.zero_()
        completed_steps = step + 1
        if (
            cfg.train.ckpt_every < max_steps
            and completed_steps % cfg.train.ckpt_every == 0
        ):
            opt_model._step = completed_steps
            save_ckpt(
                opt_model, optimizers, ckpt_path, rng=g,
                data_status=capture_data_status(completed_steps),
                extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
                barrier_group=bucket_control_group,
            )
            if rank == 0:
                print(f"[train] ckpt saved @ completed_step {completed_steps}", flush=True)
        step += 1

    if dynamic_bucket_enabled and scheduler_window["steps"]:
        flush_scheduler_window(max(start_step, step - 1), force=True)
    if os.environ.get("UV3_BENCH_NO_CKPT") != "1":
        opt_model._step = step
        save_ckpt(
            opt_model, optimizers, ckpt_path, rng=g,
            data_status=capture_data_status(step),
            extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
            barrier_group=bucket_control_group,
        )
    if rank == 0:
        lv = loss.item() if loss is not None else float("nan")
        print(f"[train] DONE step {step} loss={lv:.4f} ckpt {ckpt_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # Safe launch-time identity/length overrides keep benchmark and smoke runs
    # isolated without generating near-duplicate YAML files.
    if os.environ.get("UV3_RUN_NAME"):
        cfg.train.run_name = os.environ["UV3_RUN_NAME"]
    if os.environ.get("UV3_MAX_STEPS"):
        cfg.train.max_steps = int(os.environ["UV3_MAX_STEPS"])
    if os.environ.get("UV3_LOG_EVERY"):
        cfg.train.log_every = int(os.environ["UV3_LOG_EVERY"])
    if "UV3_MONITOR" in os.environ:
        cfg.train.monitor_enabled = os.environ["UV3_MONITOR"] == "1"
    if os.environ.get("UV3_MONITOR_NAME"):
        cfg.train.monitor_display_name = os.environ["UV3_MONITOR_NAME"]
    train(cfg)


if __name__ == "__main__":
    main()
