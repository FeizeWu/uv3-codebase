"""FSDP2 multi-GPU trainer for real pretraining (not the cached overfit).

Per-batch VAE encode + Qwen3.5 text encode (frozen, optionally node-local FSDP2) + MMDiT FSDP2 + Muon +
grad-accum + 6-stage timing + ckpt/resume (model+optimizer+RNG+data_status).
Run: torchrun --nproc_per_node=N -m uv3.train.fsdp2_trainer --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config, ExperimentConfig
from ..data.parquet_dataset import ParquetImageDataset
from ..modeling.vae import Flux2VAE
from ..modeling.qwen3_5 import Qwen3_5TextEncoder
from ..modeling.mmdit import MMDiT
from ..modeling.self_flow import build_self_flow_projector, self_flow_feature_loss, FeatureCapture, build_self_flow_latents_continuous
from ..modeling.flow import interpolate, velocity_target, logit_normal_timesteps
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
    compile_kwargs = {
        "dynamic": False,
        "mode": str(getattr(cfg.train, "compile_mode", "default")),
    }
    mmdit.compile(**compile_kwargs)
    if teacher is not None:
        teacher.compile(**compile_kwargs)
    if bool(getattr(cfg.train, "compile_text_encoder", True)):
        qwen.language_model.compile(**compile_kwargs)
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
              f"dynamic=False mode={compile_kwargs['mode']} "
              f"recompile_limit={recompile_limit}", flush=True)


def _attention_mask(mmdit, text_valid, n_img, block_size, device):
    """Build the static [1024 text, 256 image] Flex or SDPA padding mask."""
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


@torch.no_grad()
def _ema_update_local_shards_(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update an identically sharded FSDP2 teacher without materializing full params."""
    teacher_params = tuple(teacher.parameters())
    student_params = tuple(student.parameters())
    if len(teacher_params) != len(student_params):
        raise RuntimeError(
            f"EMA teacher/student parameter count mismatch: "
            f"{len(teacher_params)} != {len(student_params)}"
        )
    weight = 1.0 - decay
    for index, (teacher_param, student_param) in enumerate(zip(teacher_params, student_params)):
        teacher_local = getattr(teacher_param, "to_local", None)
        student_local = getattr(student_param, "to_local", None)
        if (teacher_local is None) != (student_local is None):
            raise RuntimeError(f"EMA parameter {index} has mismatched sharding")
        dst = teacher_local() if teacher_local is not None else teacher_param
        src = student_local() if student_local is not None else student_param
        if dst.shape != src.shape:
            raise RuntimeError(
                f"EMA local shard shape mismatch at parameter {index}: {dst.shape} != {src.shape}"
            )
        dst.lerp_(src, weight)


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


def train(cfg: ExperimentConfig):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = 0
    dev = torch.device("cuda", rank)
    dtype = torch.bfloat16
    torch.manual_seed(cfg.train.seed)

    vae, qwen, mmdit = build(cfg, dev, dtype)

    # Build the EMA teacher before FSDP so student and teacher can be sharded with
    # identical module boundaries and DTensor placements.  This lets EMA update
    # matching local shards instead of all-gathering every student parameter.
    sf_cfg = getattr(cfg.model, "self_flow", None)
    sf_enabled = bool(getattr(sf_cfg, "enabled", False)) if sf_cfg else False
    teacher = projector = student_cap = teacher_cap = None
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
        # student captures double[0] IMG stream; teacher captures single[-1] full output (slice img later)
        student_cap = FeatureCapture(stream="img_double"); student_cap.attach(mmdit.double_blocks[0])
        teacher_cap = FeatureCapture(stream="all"); teacher_cap.attach(teacher.single_blocks[-1])
        if rank == 0:
            print(f"[train] self-flow enabled coeff={sf_coeff} ema_decay={sf_decay} "
                  f"mask_ratio={sf_mask_ratio} mode={sf_timestep_mode}", flush=True)

    _compile_training_modules(vae, mmdit, qwen, teacher, cfg, rank)

    mesh = None
    if distributed and cfg.train.fsdp2:
        mesh = make_mesh(num_replicate=cfg.train.num_replicate)
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
            if os.environ.get("UV3_BENCH_LEGACY_EMA") != "1":
                apply_fsdp2(teacher, mesh, reshard_after_forward=cfg.train.reshard_after_forward)
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

    # RNG generator (defined BEFORE resume block so restore can reference it)
    g = torch.Generator(dev).manual_seed(123)

    # resume (load uses opt_model to match save's param-group keys; + RNG + data_status)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_step = 0
    resumed_epoch = 0
    if os.path.exists(ckpt_path):
        try:
            start_step = load_ckpt(opt_model, optimizers, ckpt_path)
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "rng" in ck:
                torch.set_rng_state(ck["rng"]["py"])
                if torch.cuda.is_available() and "cuda" in ck["rng"]:
                    torch.cuda.set_rng_state(ck["rng"]["cuda"])
                if "py_gen" in ck["rng"]:
                    g.set_state(ck["rng"]["py_gen"])
            resumed_epoch = ck.get("data_status", {}).get("epoch", 0)
            if rank == 0:
                print(f"[train] resumed from step {start_step} (epoch {resumed_epoch})", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[train] resume failed ({type(e).__name__}: {e}); starting fresh", flush=True)

    # data: TarMetadataDataset (real data) or ParquetImageDataset (imagenet smoke)
    if cfg.data.dataset == "tar":
        from ..data.tar_dataset import TarMetadataDataset
        ds = TarMetadataDataset(
            manifest_path=cfg.data.root,  # manifest path stored in root
            image_size=cfg.data.image_size,
            caption_field=getattr(cfg.data, "caption_field", "caption_qwen3_7_flash"),
        )
        ds.set_epoch(resumed_epoch)
        if resumed_epoch > 0 or hasattr(ds, "_resume_shard") and ds._resume_shard > 0:
            ds.set_resume(resumed_epoch, 0)  # coarse: epoch-level resume
    else:
        ds = ParquetImageDataset(
            root=cfg.data.root, split=cfg.data.split, parquet_glob=cfg.data.parquet_glob,
            image_size=cfg.data.image_size, image_field=cfg.data.image_field,
            label_field=getattr(cfg.data, "label_field", "label"),
        )
        ds.set_epoch(resumed_epoch)
    text_buckets = tuple(int(x) for x in getattr(cfg.train, "text_length_buckets", ()))
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
                f"weights={text_bucket_weights}",
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
    measured_steps = 0
    bucket_counts = {}
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
        with torch.no_grad():
            latents = vae.encode_images(batch["pixel_values"])      # vae
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
            if bucket_length < cfg.model.qwen_vl.max_length:
                pad_length = cfg.model.qwen_vl.max_length - bucket_length
                text = F.pad(text, (0, 0, 0, pad_length))
                mask = F.pad(mask, (0, pad_length), value=0)
            n_img = (latents.shape[-1] // 2) * (latents.shape[-2] // 2)
            expected_img = (cfg.data.image_size // 16) ** 2
            if text.shape[1] != cfg.model.qwen_vl.max_length:
                raise RuntimeError(
                    f"static compile expects {cfg.model.qwen_vl.max_length} text tokens, "
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
        timer.mark("qwen")
        if sf_enabled:
            t = logit_normal_timesteps(bs, dev)
            noise = torch.randn_like(latents)
            # per-token self-flow: mixed student latents + per-token timesteps
            student_latents, teach_lat, t_teach, token_t = build_self_flow_latents_continuous(
                latents, noise, t,
                mask_ratio=sf_mask_ratio, ratio=sf_ratio, timestep_mode=sf_timestep_mode,
            )
            # student forward with per-token timestep modulation (captures double[0] img stream)
            pred_v = mmdit(student_latents, text, t,
                           token_timesteps=token_t, text_attn_mask=text_attn_mask)
            fm_loss = F.mse_loss(pred_v.float(), velocity_target(latents, noise).float())
            with torch.no_grad():
                _ = teacher(teach_lat, text, t_teach, text_attn_mask=text_attn_mask)
            s_feat = student_cap.features
            t_feat = teacher_cap.features
            # teacher returns full [txt, img] sequence; slice img segment dynamically
            n_txt_actual = text.shape[1]
            t_feat_img = t_feat[:, n_txt_actual:]
            n_tok = min(s_feat.shape[1], t_feat_img.shape[1])
            proj_s = projector(s_feat[:, :n_tok])
            feat_loss = self_flow_feature_loss(t_feat_img[:, :n_tok], proj_s)
            loss = fm_loss + sf_coeff * feat_loss
        else:
            loss = mmdit.training_loss(latents, text, text_attn_mask=text_attn_mask)  # basic-FM forward
        timer.mark("forward")
        (loss / accum).backward()                                     # backward
        timer.mark("backward")
        if (step + 1) % accum == 0:
            # clip mmdit (FSDP2 DTensor) and projector (regular) SEPARATELY (mixed types break foreach)
            torch.nn.utils.clip_grad_norm_(mmdit.parameters(), cfg.train.grad_clip)
            if projector is not None:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), cfg.train.grad_clip)
            for o in optimizers.values():
                o.step()                                             # optimizer
            for o in optimizers.values():
                o.zero_grad()
            # EMA matching local shards. Student and teacher use the same FSDP2
            # wrapping, so this is mathematically identical to full-parameter EMA
            # without a per-parameter all-gather or a replicated teacher.
            if sf_enabled:
                if os.environ.get("UV3_BENCH_LEGACY_EMA") == "1":
                    with torch.no_grad():
                        for teacher_param, student_param in zip(teacher.parameters(), mmdit.parameters()):
                            full_tensor = getattr(student_param, "full_tensor", None)
                            teacher_param.lerp_(
                                full_tensor() if full_tensor is not None else student_param,
                                1 - sf_decay,
                            )
                else:
                    _ema_update_local_shards_(teacher, mmdit, sf_decay)
        timer.mark("step")
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
        if rank == 0 and step % log_every == 0:
            spd = measured_steps / (time.time() - t0 + 1e-9) if measured_steps else 0.0
            ts = timer.summary()
            mem_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            mem_pct = 100.0 * torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(dev).total_memory
            timing_str = " ".join(f"{k}={v:.3f}s" for k, v in ts.items()) if ts else ""
            bucket_str = f" buckets={dict(sorted(bucket_counts.items()))}" if text_buckets else ""
            print(f"[train] step {step:5d} loss={loss.item():.4f} {spd:.2f}it/s mem={mem_gib:.1f}GiB({mem_pct:.1f}%){bucket_str} [{timing_str}]", flush=True)
            # JSONL metrics: timing_unit 自文档,说明 timing 是窗口和而非每步
            import json as _json
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as _jf:
                _json.dump({"step": step, "loss": loss.item(), "spd": spd,
                            "max_memory_reserved_gib": mem_gib, "max_memory_reserved_pct": mem_pct,
                            "timing": ts, "timing_unit": f"sum_over_{log_every}_steps",
                            "text_bucket_counts": dict(sorted(bucket_counts.items()))}, _jf)
                _jf.write("\n")
            timer.reset()
            bucket_counts = {}
        if cfg.train.ckpt_every < max_steps and step > 0 and step % cfg.train.ckpt_every == 0:
            opt_model._step = step
            save_ckpt(opt_model, optimizers, ckpt_path, rng=g,
                      data_status={"epoch": step // 1000, "step": step})
            if rank == 0:
                print(f"[train] ckpt saved @ step {step}", flush=True)
        step += 1

    if os.environ.get("UV3_BENCH_NO_CKPT") != "1":
        opt_model._step = step
        save_ckpt(opt_model, optimizers, ckpt_path, rng=g,
                  data_status={"epoch": step // 1000, "step": step})
    if rank == 0:
        lv = loss.item() if loss is not None else float("nan")
        print(f"[train] DONE step {step} loss={lv:.4f} ckpt {ckpt_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
