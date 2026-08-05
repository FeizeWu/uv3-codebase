"""FSDP2 multi-GPU trainer for real pretraining (not the cached overfit).

Per-batch VAE encode + Qwen3.5 text encode (frozen, optionally node-local FSDP2) + MMDiT FSDP2 + Muon +
grad-accum + 6-stage timing + ckpt/resume (model+optimizer+RNG+data_status).
Run: torchrun --nproc_per_node=N -m uv3.train.fsdp2_trainer --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import shutil
import time
from dataclasses import asdict

import torch
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
)
from ..modeling.flow import interpolate, velocity_target
from ..data.noise_scheduler import sample_timesteps, timestep_bin_sums
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
    """Compile fixed-shape MMDiT calls and the frozen Qwen backbone before FSDP2."""
    if not cfg.train.compile:
        return
    text_buckets = tuple(int(x) for x in getattr(cfg.train, "text_length_buckets", ()))
    pad_text_to_max = bool(getattr(cfg.train, "pad_text_to_max_length", True))
    recompile_limit = "default"
    if text_buckets:
        # Qwen's hybrid linear/full-attention stack creates more than one guarded
        # frame per sequence length (including bool/long internal mask variants).
        # The default limit of 8 makes a five-bucket run silently fall back to
        # eager mode, so reserve enough cache entries for all static graphs.
        from torch import _dynamo
        _dynamo.config.recompile_limit = max(
            int(_dynamo.config.recompile_limit),
            16 * len(text_buckets),
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
        # AutoencoderKLFlux2.encode() is decorated and wraps its tensor result in
        # a posterior object. Compile the fixed-shape tensor-only core instead.
        vae.vae._encode = torch.compile(
            vae.vae._encode,
            dynamic=False,
            mode=str(getattr(cfg.train, "vae_compile_mode", "default")),
        )
    if rank == 0:
        print(f"[train] torch.compile ON: mmdit=True teacher={teacher is not None} "
              f"text_encoder={bool(getattr(cfg.train, 'compile_text_encoder', True))} "
              f"vae={compile_vae} "
              f"vae_mode={str(getattr(cfg.train, 'vae_compile_mode', 'default')) if compile_vae else 'off'} "
              f"mmdit_text_buckets={bool(text_buckets) and not pad_text_to_max} "
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

    def __init__(self, loader, dev, dtype):
        self.loader = iter(loader)
        self.dev = dev
        self.dtype = dtype
        self.stream = torch.cuda.Stream()
        self._next = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            pv = batch["pixel_values"].to(self.dev, non_blocking=True).to(self.dtype)
            self._next = {"pixel_values": pv, "text": batch["text"]}
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

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        out = self._next
        self._preload()
        return out


class LengthBucketBatcher:
    """Form full batches with one static text length, identically scheduled per rank."""

    def __init__(self, loader, tokenizer, batch_size, buckets, weights):
        self.loader = loader
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.buckets = tuple(int(x) for x in buckets)
        self.weights = tuple(int(x) for x in weights)
        self._slot = 0
        self._token_buffers = {}
        if not self.buckets or tuple(sorted(self.buckets)) != self.buckets:
            raise ValueError(f"text_length_buckets must be sorted and non-empty: {self.buckets}")
        if len(self.weights) != len(self.buckets) or any(x < 1 for x in self.weights):
            raise ValueError(
                "text_length_bucket_weights must contain one positive integer "
                f"per bucket: buckets={self.buckets}, weights={self.weights}"
            )

    def _tokenize_for_bucket(self, text):
        token_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.buckets[-1],
        )["input_ids"]
        length = len(token_ids)
        bucket = next((bucket for bucket in self.buckets if length <= bucket), self.buckets[-1])
        return bucket, torch.tensor(token_ids, dtype=torch.long)

    def __iter__(self):
        queues = {bucket: [] for bucket in self.buckets}
        schedule = [
            bucket
            for bucket, weight in zip(self.buckets, self.weights)
            for _ in range(weight)
        ]
        # Same seed on every rank keeps the bucket/compile-graph order identical,
        # while shuffling avoids long runs of one bucket filling other queues.
        random.Random(0).shuffle(schedule)
        targets = itertools.cycle(schedule)
        source = iter(self.loader)
        while True:
            target = next(targets)
            while len(queues[target]) < self.batch_size:
                try:
                    sample = next(source)
                except StopIteration:
                    return
                bucket, input_ids = self._tokenize_for_bucket(sample["text"])
                sample["input_ids"] = input_ids
                queues[bucket].append(sample)
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
            token_key = (slot, target)
            buffers = self._token_buffers.get(token_key)
            if buffers is None:
                buffers = (
                    torch.empty(
                        (self.batch_size, target),
                        dtype=torch.long,
                        pin_memory=pin_memory,
                    ),
                    torch.empty(
                        (self.batch_size, target),
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
            yield {
                "pixel_values": pixel_values,
                "text": [sample["text"] for sample in samples],
                "text_bucket_length": target,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


def build(cfg: ExperimentConfig, dev, dtype):
    vae = Flux2VAE.from_pretrained(cfg.model.vae.pretrained, dtype=dtype, force_upcast=False).to(dev).eval()
    qwen = Qwen3_5TextEncoder.from_pretrained(
        cfg.model.qwen_vl.pretrained, max_length=cfg.model.qwen_vl.max_length, dtype=dtype
    ).to(dev).eval()
    cfg.model.flex_attention = bool(cfg.model.flex_attention or cfg.train.flex_attention)
    mmdit = MMDiT.build(cfg.model.transformer, cfg.model, text_encoder=qwen).to(dev, dtype=dtype)
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
    sf_coeff = 0.0
    sf_decay = 0.99
    sf_n_txt = 64
    sf_mask_ratio = 0.5
    sf_ratio = 0.5
    sf_timestep_mode = "ratio"
    if sf_enabled:
        teacher = copy.deepcopy(mmdit).to(dev, dtype=dtype)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        projector = build_self_flow_projector(mmdit.inner_dim).to(dev, dtype=dtype)
        sf_coeff = float(getattr(sf_cfg, "coeff", 1.0))
        sf_decay = float(getattr(sf_cfg, "ema_decay", 0.9999))
        sf_n_txt = int(getattr(sf_cfg, "n_txt", 64))
        sf_mask_ratio = float(getattr(sf_cfg, "mask_ratio", 0.5))
        sf_ratio = float(getattr(sf_cfg, "ratio", 0.5))
        sf_timestep_mode = str(getattr(sf_cfg, "timestep_mode", "ratio"))
        student_cap, teacher_cap, sf_student_has_text_prefix = (
            attach_self_flow_feature_captures(
                mmdit,
                teacher,
                student_depth=int(getattr(sf_cfg, "student_depth", -1)),
                teacher_depth=int(getattr(sf_cfg, "teacher_depth", -1)),
            )
        )
        if rank == 0:
            print(f"[train] self-flow enabled coeff={sf_coeff} ema_decay={sf_decay} "
                  f"mask_ratio={sf_mask_ratio} mode={sf_timestep_mode}", flush=True)

    # Model/projector initialization must match on every rank, while training
    # noise and timestep draws must not be identical across data-parallel ranks.
    torch.manual_seed(cfg.train.seed + rank)

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
    start_step = 0
    resumed_epoch = 0
    if os.path.exists(ckpt_path):
        try:
            start_step, resume_payload = load_ckpt(
                opt_model, optimizers, ckpt_path,
                extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
                return_payload=True,
            )
            ck = resume_payload
            rank_rngs = ck.get("rng_by_rank")
            restored_rng = (
                rank_rngs[rank]
                if rank_rngs is not None and rank < len(rank_rngs)
                else ck.get("rng")
            )
            if restored_rng is not None:
                torch.set_rng_state(restored_rng["py"])
                if torch.cuda.is_available() and "cuda" in restored_rng:
                    torch.cuda.set_rng_state(restored_rng["cuda"])
                if "py_gen" in restored_rng:
                    g.set_state(restored_rng["py_gen"])
            resumed_epoch = ck.get("data_status", {}).get("epoch", 0) or 0
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
                print(
                    "[train] WARNING: model/optimizer/per-rank RNG are exact, but "
                    "the multi-worker text-bucket data cursor restarts at the saved epoch",
                    flush=True,
                )
        except Exception as e:
            if rank == 0:
                print(f"[train] resume failed ({type(e).__name__}: {e})", flush=True)
            raise RuntimeError(f"refusing to start fresh after checkpoint load failure: {ckpt_path}") from e

    # data: TarMetadataDataset (real data) or ParquetImageDataset (imagenet smoke)
    if cfg.data.dataset == "tar":
        from ..data.tar_dataset import TarMetadataDataset
        ds = TarMetadataDataset(
            manifest_path=cfg.data.root,  # manifest path stored in root
            image_size=cfg.data.image_size,
            caption_field=getattr(cfg.data, "caption_field", "caption_qwen3_7_flash"),
        )
        ds.set_epoch(resumed_epoch)
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
        )
        ds.set_epoch(resumed_epoch)
    else:
        ds = ParquetImageDataset(
            root=cfg.data.root, split=cfg.data.split, parquet_glob=cfg.data.parquet_glob,
            image_size=cfg.data.image_size, image_field=cfg.data.image_field,
            label_field=getattr(cfg.data, "label_field", "label"),
        )
        ds.set_epoch(resumed_epoch)
    text_buckets = tuple(int(x) for x in getattr(cfg.train, "text_length_buckets", ()))
    pad_text_to_max = bool(getattr(cfg.train, "pad_text_to_max_length", True))
    if text_buckets:
        text_bucket_weights = tuple(
            int(x) for x in getattr(cfg.train, "text_length_bucket_weights", ())
        )
        sample_loader = DataLoader(ds, batch_size=None, num_workers=cfg.data.num_workers)
        loader = LengthBucketBatcher(
            sample_loader,
            qwen.tokenizer,
            cfg.train.batch_size_per_gpu,
            text_buckets,
            text_bucket_weights,
        )
        if rank == 0:
            print(
                f"[train] static Qwen length buckets={text_buckets} "
                f"weights={text_bucket_weights} "
                f"mmdit_text_buckets={not pad_text_to_max}",
                flush=True,
            )
    else:
        loader = DataLoader(
            ds,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.data.num_workers,
            drop_last=True,
        )
    prefetcher = DataPrefetcher(loader, dev, dtype)
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
    timestep_metrics = bool(getattr(cfg.train, "timestep_metrics", False))
    timestep_bin_counts = torch.zeros(10, device=dev, dtype=torch.float64)
    timestep_bin_fm_sums = torch.zeros_like(timestep_bin_counts)
    timestep_bin_sf_sums = torch.zeros_like(timestep_bin_counts)
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
            expected_img = (cfg.data.image_size // 16) ** 2
            expected_text = cfg.model.qwen_vl.max_length if pad_text_to_max else bucket_length
            if text.shape[1] != expected_text:
                raise RuntimeError(
                    f"static compile expects {expected_text} text tokens, "
                    f"got {text.shape[1]}"
                )
            if n_img != expected_img:
                raise RuntimeError(
                    f"static compile expects {expected_img} image tokens from "
                    f"{cfg.data.image_size}x{cfg.data.image_size}, got {n_img}"
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
        if sf_enabled:
            # per-token self-flow: mixed student latents + per-token timesteps
            student_latents, teach_lat, t_teach, token_t = build_self_flow_latents_continuous(
                latents, noise, t,
                mask_ratio=sf_mask_ratio, ratio=sf_ratio, timestep_mode=sf_timestep_mode,
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
            t_feat_img = t_feat[:, n_txt_actual:]
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
            loss = fm_loss_per_sample.mean()
            profile_mark("student_forward_and_fm_loss")
        if timestep_metrics:
            bin_counts, bin_fm, bin_sf = timestep_bin_sums(
                t, fm_loss_per_sample, sf_loss_per_sample,
            )
            timestep_bin_counts += bin_counts
            timestep_bin_fm_sums += bin_fm
            timestep_bin_sf_sums += bin_sf
        timer.mark("forward")
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
                    mmdit.parameters(), cfg.train.grad_clip
                )
                if audit_grad_norm:
                    # Execute on every rank: DTensor scalar materialization may
                    # participate in collectives even though only rank 0 records it.
                    grad_norm_value = float(mmdit_grad_norm.detach().item())
                    if rank == 0:
                        grad_norm_samples.append(grad_norm_value)
                if projector is not None:
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), cfg.train.grad_clip)
            profile_mark("grad_clip")
            for o in optimizers.values():
                o.step()                                             # optimizer
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
        if rank == 0 and step % log_every == 0:
            spd = measured_steps / (time.time() - t0 + 1e-9) if measured_steps else 0.0
            ts = timer.summary()
            mem_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            mem_pct = 100.0 * torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(dev).total_memory
            timing_str = " ".join(f"{k}={v:.3f}s" for k, v in ts.items()) if ts else ""
            bucket_str = f" buckets={dict(sorted(bucket_counts.items()))}" if text_buckets else ""
            grad_norm_stats = None
            grad_norm_str = ""
            timestep_stats = None
            if reduced_timestep_bins is not None:
                counts, fm_sums, sf_sums = reduced_timestep_bins
                safe_counts = counts.clamp_min(1.0)
                timestep_stats = {
                    "edges": [i / 10 for i in range(11)],
                    "count": counts.to(torch.int64).tolist(),
                    "fm_loss": (fm_sums / safe_counts).tolist(),
                    "self_flow_loss": (sf_sums / safe_counts).tolist() if sf_enabled else None,
                }
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
            print(f"[train] step {step:5d} loss={loss.item():.4f} {spd:.2f}it/s mem={mem_gib:.1f}GiB({mem_pct:.1f}%){bucket_str}{grad_norm_str} [{timing_str}]", flush=True)
            # JSONL metrics: timing_unit 自文档,说明 timing 是窗口和而非每步
            import json as _json
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as _jf:
                _json.dump({"step": step, "loss": loss.item(), "spd": spd,
                            "world_size": world_size,
                            "global_batch_size": bs * world_size * accum,
                            "max_memory_reserved_gib": mem_gib, "max_memory_reserved_pct": mem_pct,
                            "timing": ts, "timing_unit": f"sum_over_{log_every}_steps",
                            "text_bucket_counts": dict(sorted(bucket_counts.items())),
                            "grad_norm": grad_norm_stats,
                            "timestep_bins": timestep_stats}, _jf)
                _jf.write("\n")
            timer.reset()
            bucket_counts = {}
            grad_norm_samples = []
        if timestep_metrics and step % log_every == 0:
            timestep_bin_counts.zero_()
            timestep_bin_fm_sums.zero_()
            timestep_bin_sf_sums.zero_()
        completed_steps = step + 1
        if (
            cfg.train.ckpt_every < max_steps
            and completed_steps % cfg.train.ckpt_every == 0
        ):
            opt_model._step = completed_steps
            save_ckpt(
                opt_model, optimizers, ckpt_path, rng=g,
                data_status={"epoch": resumed_epoch, "step": completed_steps},
                extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
            )
            if rank == 0:
                print(f"[train] ckpt saved @ completed_step {completed_steps}", flush=True)
        step += 1

    if os.environ.get("UV3_BENCH_NO_CKPT") != "1":
        opt_model._step = step
        save_ckpt(
            opt_model, optimizers, ckpt_path, rng=g,
            data_status={"epoch": resumed_epoch, "step": step},
            extra_models={"self_flow_teacher": teacher} if sf_enabled else None,
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
