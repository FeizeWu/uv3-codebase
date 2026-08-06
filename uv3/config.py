"""Experiment configuration: YAML + nested dataclasses (UniWorld-style, extended).

Note: NO `from __future__ import annotations` — we need f.type to resolve to actual
dataclass classes (not strings) for _construct/_field_dataclass is_dataclass() checks.
Python 3.12 natively supports `str | None` / `tuple[int, ...]` annotations.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any
import yaml


@dataclass
class ComponentConfig:
    """A loadable component (vae / qwen_vl / transformer)."""
    backend: str = "transformers"          # transformers | diffusers | random | none
    pretrained: str | None = None          # local path or modelscope id
    subfolder: str | None = None
    trainable: bool = False
    frozen: bool = False                  # explicit freeze (no grad)
    # 文本编码器截断上限:仅 qwen_vl 消费(vae/transformer 忽略)。
    # 必须是 dataclass 字段才能被 _construct 读入——否则 YAML 里写 max_length 会被静默丢弃,
    # trainer 就被迫硬编码(历史 bug:max_length=64 覆盖了 qwen3_5.py 默认 1024)。
    max_length: int = 1024


@dataclass
class SelfFlowConfig:
    enabled: bool = True
    coeff: float = 1.0                     # projection-loss weight
    teacher_depth: int = -1               # teacher single-block index (-1 = last)
    student_depth: int = -1               # first available stream index (-1 = first)
    timestep_mode: str = "ratio"          # ratio | random_cleaner | min
    ratio: float = 0.5                    # paired_t = t * ratio (for ratio mode)
    mask_ratio: float = 0.5               # per-token timestep masking probability
    ema_decay: float = 0.9999
    projector_dim: int | None = None      # None => 2*hidden
    n_txt: int = 64                       # text token count (for img segment slicing)


@dataclass
class ModelConfig:
    architecture: str = "mmdit"            # mmdit (dual/single) — Qwen-Image-style
    hidden_size: int = 1536
    num_layers: int = 12
    num_double_layers: int | None = None   # dual-stream count (None => num_layers)
    num_single_layers: int | None = None   # single-stream count (None => num_layers)
    num_heads: int = 12
    head_dim: int | None = None
    latent_channels: int = 32              # FLUX.2 VAE z=32
    patch_size: int = 2                    # 2x2 vec-pack
    in_channels: int = 4                   # RGB+alpha slot (FLUX.2 VAE in=4)
    out_channels: int = 4
    rope_theta: float = 2_000.0
    axes_dims_rope: tuple[int, ...] = (16, 16, 16, 16)
    guidance_embeds: bool = False
    flex_attention: bool = False            # use FlexAttention processors in the MMDiT
    alpha_on: bool = True                  # 4th channel active (dummy now)
    self_flow: SelfFlowConfig = field(default_factory=SelfFlowConfig)
    vae: ComponentConfig = field(default_factory=ComponentConfig)
    qwen_vl: ComponentConfig = field(default_factory=ComponentConfig)
    transformer: ComponentConfig = field(default_factory=ComponentConfig)


@dataclass
class DataConfig:
    dataset: str = "parquet"               # parquet streaming
    root: str = "/mnt/oss/users/lzj/imagenet-ablation/imagenet-1k"
    split: str = "train"
    parquet_glob: str = "data/train-*.parquet"
    image_field: str = "image"
    caption_field: str = "text"
    image_size: int = 256
    bucket: bool = False                   # group batches by source aspect ratio
    # Equal-area profiles from uv3.data.bucket_sampler. ``mar_256`` remains an
    # alias for this five-bucket preset for old configs.
    aspect_buckets: tuple[str, ...] = (
        "square", "landscape", "portrait", "widescreen", "phone",
    )
    aspect_bucket_weights: tuple[int, ...] = ()
    resolution_stride: int = 16
    alpha: bool = True                     # preserve RGBA else pad ones
    it2i_mix: float = 0.0                  # 0 = pure t2i; >0 = fraction of edit pairs
    it2i_parquet: str | None = None
    num_workers: int = 4
    shuffle_shards: bool = True
    overfit_n: int | None = None           # cached overfit set size
    label_field: str = "label"


@dataclass
class OptimizerConfig:
    optimizer: str = "muon_adam"           # muon_adam | adamw
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_rms_scale: bool = True
    muon_weight_decay: float = 0.01
    adam_lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-15
    adam_weight_decay: float = 0.01
    adam_fused: bool = False


@dataclass
class TrainConfig:
    fsdp2: bool = True
    # Shard the frozen text backbone only inside each node.  This keeps Qwen
    # collectives on NVLink/NVSwitch even when the training mesh spans nodes.
    fsdp_text_encoder: bool = False
    text_encoder_shard_size: int = 8
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD | SHARD_GRAD_OP | HYBRID_SHARD
    num_replicate: int = 1                  # HSDP 2D mesh replicate dim
    num_shard: int | None = None            # None => world_size // num_replicate
    mixed_precision: str = "bf16"
    reshard_after_forward: bool = True
    compile: bool = True
    compile_text_encoder: bool = True       # compile frozen Qwen language backbone too
    text_encoder_fp8: bool = False          # torchao tensorwise FP8 dynamic-act/FP8-weight
    text_encoder_fp8_scope: str = "all"     # all | mlp | mlp_middle
    text_encoder_fp8_mlp_start: int = 4      # inclusive for mlp_middle
    text_encoder_fp8_mlp_end: int = 28       # exclusive for mlp_middle
    compile_vae: bool = False               # compile fixed-shape frozen VAE encoder
    compile_dynamic: bool = False           # fixed 1024 text + 256 image tokens
    compile_mode: str = "default"           # default | reduce-overhead | max-autotune[-no-cudagraphs]
    mmdit_fp8: bool = False                 # torchao FP8 training for token-block Linear layers
    # Communicate trainable FP8 weights directly during FSDP all-gather instead
    # of expanding their local shards to BF16 first.  Requires mmdit_fp8 + FSDP2.
    mmdit_fp8_fsdp_all_gather: bool = False
    # Pack all dynamic FP8 weight scales into one all-reduce after optimizer.step.
    # If false, torchao computes/reduces a scale in each FSDP pre-all-gather.
    mmdit_fp8_precompute_scale: bool = True
    # None inherits compile_mode for backward compatibility. Set explicitly to
    # benchmark MMDiT compiler modes without changing the frozen Qwen graphs.
    text_encoder_compile_mode: str | None = None
    vae_compile_mode: str = "default"
    # Optional static Qwen sequence-length buckets.  Every rank follows the same
    # weighted schedule so distributed training never waits on a longer bucket
    # chosen independently by another rank.
    text_length_buckets: tuple[int, ...] = ()
    text_length_bucket_weights: tuple[int, ...] = ()
    # Keep each Qwen bucket length through MMDiT instead of padding every text
    # embedding back to qwen_vl.max_length. This creates one static MMDiT graph
    # per configured bucket and removes work on masked text padding.
    pad_text_to_max_length: bool = True
    flex_attention: bool = True
    block_size: int = 128
    batch_size_per_gpu: int = 16
    grad_accum: int = 1
    max_steps: int = 100
    lr: float = 0.0                         # informational; optimizer lrs in OptimizerConfig
    warmup_steps: int = 0
    lr_schedule: str = "constant"           # constant | cosine
    grad_clip: float = 1.0
    # Clip every step by default. With a positive warmup and interval=0,
    # clipping runs only during warmup; interval=N checks every N steps after it.
    grad_clip_warmup_steps: int = 0
    grad_clip_interval: int = 1
    ckpt_every: int = 2000
    output_dir: str = "/mnt/oss/users/wfz/uv3-codebase-runs"
    run_name: str = "run"
    wandb: bool = False
    wandb_project: str = "uv3"
    monitor_enabled: bool = False
    monitor_display_name: str | None = None
    profile: bool = False
    log_every: int = 10
    seed: int = 42
    # Flow convention is always t=0 clean, t=1 noise. Sampling never flips t.
    timestep_strategy: str = "logit_normal"  # uniform | logit_normal | logit_normal_shift
    timestep_logit_mean: float = 0.0
    timestep_logit_std: float = 1.0
    timestep_shift: float | None = None       # None => dynamic shift for *_shift
    timestep_base_seq_len: int = 256
    timestep_max_seq_len: int = 8192
    timestep_base_shift: float = 0.5
    timestep_max_shift: float = 0.9
    timestep_metrics: bool = False            # 10-bin distributed online statistics
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _construct(cls, raw: dict[str, Any]):
    """Recursively build a dataclass from a dict, tolerating extra keys."""
    if not is_dataclass(cls):
        return raw
    kwargs = {}
    raw = raw or {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        ft = f.type
        # nested dataclass field?
        if is_dataclass(ft) and isinstance(val, dict):
            kwargs[f.name] = _construct(ft, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str) -> ExperimentConfig:
    raw = yaml.safe_load(open(path).read()) or {}
    model_raw = raw.get("model", {}) or {}
    for name in ("vae", "qwen_vl", "transformer", "self_flow"):
        if name in model_raw:
            model_raw[name] = _construct(
                _field_dataclass(ModelConfig, name), model_raw[name]
            )
    cfg = ExperimentConfig(
        model=_construct(ModelConfig, model_raw),
        data=_construct(DataConfig, raw.get("data", {})),
        train=_construct(TrainConfig, raw.get("train", {})),
    )
    return cfg


def _field_dataclass(parent_cls, field_name: str):
    for f in fields(parent_cls):
        if f.name == field_name and is_dataclass(f.type):
            return f.type
    raise KeyError(f"{field_name} is not a nested dataclass field of {parent_cls}")
