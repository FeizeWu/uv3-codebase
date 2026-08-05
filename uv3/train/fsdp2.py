"""FSDP2 native wrapper (fully_shard). Pattern from transfusion-core fsdp2_utils_mot.

Shards MMDiT transformer blocks (leaf) then root. Native torch.optim.Muon works under
FSDP2 (review's 2-GPU test: cos~0.99997 via DTensor implicit collectives; NOT officially
supported but empirically correct on torch 2.12). ckpt via checkpoint.state_dict.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh


def resolve_mesh_shape(
    world_size: int,
    num_replicate: int = 1,
    num_shard: int | None = None,
) -> tuple[int, ...]:
    """Resolve a 1D FSDP or 2D HSDP mesh without initializing distributed state.

    ``num_shard`` takes precedence so an 8-GPU-node config stays node-local as
    world size grows: world=256, num_shard=8 -> (replicate=32, shard=8).
    ``num_replicate`` remains supported for older configs.
    """
    world_size = int(world_size)
    num_replicate = int(num_replicate)
    if world_size < 1 or num_replicate < 1:
        raise ValueError("world_size and num_replicate must be positive")
    if num_shard is not None:
        num_shard = int(num_shard)
        if num_shard < 1 or world_size % num_shard:
            raise ValueError(
                f"WORLD_SIZE={world_size} must be divisible by num_shard={num_shard}"
            )
        resolved_replicate = world_size // num_shard
        if num_replicate not in (1, resolved_replicate):
            raise ValueError(
                f"conflicting mesh: num_replicate={num_replicate}, "
                f"num_shard={num_shard}, WORLD_SIZE={world_size}"
            )
        num_replicate = resolved_replicate
    else:
        if world_size % num_replicate:
            raise ValueError(
                f"WORLD_SIZE={world_size} must be divisible by "
                f"num_replicate={num_replicate}"
            )
        num_shard = world_size // num_replicate
    return (world_size,) if num_replicate == 1 else (num_replicate, num_shard)


def make_mesh(
    device_type: str = "cuda",
    num_replicate: int = 1,
    num_shard: int | None = None,
    world_size: int | None = None,
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    mesh_shape = resolve_mesh_shape(world_size, num_replicate, num_shard)
    if len(mesh_shape) == 1:
        return init_device_mesh(device_type, mesh_shape, mesh_dim_names=("shard",))
    return init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )


def make_node_local_mesh(device_type: str = "cuda", shard_size: int | None = None):
    """Return the current rank's node-local Qwen shard mesh.

    The shard size may be smaller than LOCAL_WORLD_SIZE (for example 2 or 4 on
    an 8-GPU node).  Contiguous local ranks are grouped together, and the
    divisibility check guarantees that no group crosses a node boundary.
    """
    world_size = torch.distributed.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if local_world_size < 1 or world_size % local_world_size:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )
    shard_size = local_world_size if shard_size is None else int(shard_size)
    if shard_size < 1 or local_world_size % shard_size:
        raise ValueError(
            f"LOCAL_WORLD_SIZE={local_world_size} must be divisible by "
            f"text_encoder_shard_size={shard_size}"
        )
    num_groups = world_size // shard_size
    mesh = init_device_mesh(
        device_type,
        (num_groups, shard_size),
        mesh_dim_names=("qwen_replicate", "qwen_shard"),
    )
    return mesh["qwen_shard"]


def apply_frozen_text_encoder_fsdp2(
    text_encoder: torch.nn.Module,
    local_mesh,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
):
    """Shard a frozen Qwen backbone layer-by-layer on the node-local mesh.

    Layer-level wrapping is important: wrapping the whole 7.9B backbone as one
    FSDP unit would all-gather all weights at once and erase most peak-memory
    savings.  Frozen parameters need forward all-gathers but no gradient reduce.
    """
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    backbone = getattr(text_encoder, "language_model", text_encoder)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError(
            f"unsupported text backbone {type(backbone).__name__}: missing .layers"
        )
    for layer in layers:
        fully_shard(
            layer,
            mesh=local_mesh,
            mp_policy=mp,
            reshard_after_forward=True,
        )
    for name in ("embed_tokens", "norm"):
        submodule = getattr(backbone, name, None)
        if submodule is not None and list(submodule.parameters()):
            fully_shard(
                submodule,
                mesh=local_mesh,
                mp_policy=mp,
                reshard_after_forward=True,
            )
    if list(backbone.parameters(recurse=False)):
        fully_shard(
            backbone,
            mesh=local_mesh,
            mp_policy=mp,
            reshard_after_forward=True,
        )
    return text_encoder


def apply_fsdp2(
    model: torch.nn.Module,
    dp_mesh,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    reshard_after_forward: bool = True,
):
    """Apply fully_shard bottom-up: leaf blocks, then root-level submodules individually.

    Key (round 5): when forward_per_token() is used (self-flow), transformer.forward() is NOT
    called — submodules are invoked directly. So we must shard each root-level submodule
    individually (x_embedder, context_embedder, time_guidance_embed, modulation modules,
    norm_out, proj_out) instead of wrapping the whole transformer root. Block-level shards
    still unshard correctly when we call block() directly.
    """
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    t = getattr(model, "transformer", model)
    # shard leaf blocks (these unshard on block.forward() — called directly in forward_per_token)
    for blk in list(getattr(t, "transformer_blocks", [])):
        fully_shard(blk, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    for blk in list(getattr(t, "single_transformer_blocks", [])):
        fully_shard(blk, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    # shard root-level submodules individually (so they unshard when called directly)
    for name in ["x_embedder", "context_embedder", "time_guidance_embed",
                 "double_stream_modulation_img", "double_stream_modulation_txt",
                 "single_stream_modulation", "norm_out", "proj_out"]:
        sub = getattr(t, name, None)
        if sub is not None and list(sub.parameters()):
            fully_shard(sub, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    # shard the wrapper if it has own params
    if list(model.parameters(recurse=False)):
        fully_shard(model, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    return model


def save_ckpt(model, optimizers: dict, path: str, rng=None, data_status=None, extra_models=None):
    """Save model + optimizer + RNG + data_status. Works in dist (rank0+barrier) and single modes."""
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
    )
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    msd = get_model_state_dict(model, options=opts)
    osd = {k: get_optimizer_state_dict(model, o, options=opts) for k, o in optimizers.items()}
    extra_model_states = {
        name: get_model_state_dict(extra_model, options=opts)
        for name, extra_model in (extra_models or {}).items()
    }
    rng_state = {"py": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
    if rng is not None and hasattr(rng, "get_state"):
        rng_state["py_gen"] = rng.get_state()
    dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank0 = (not dist_ok) or torch.distributed.get_rank() == 0
    if dist_ok:
        rng_by_rank = [None] * torch.distributed.get_world_size() if rank0 else None
        torch.distributed.gather_object(rng_state, rng_by_rank, dst=0)
    else:
        rng_by_rank = [rng_state]
    if rank0:
        payload = {"model": msd, "optim": osd, "step": getattr(model, "_step", 0),
                   "rng": rng_state, "rng_by_rank": rng_by_rank,
                   "data_status": data_status,
                   "extra_models": extra_model_states}
        target = Path(path)
        staging_root = Path(os.environ.get(
            "UV3_CKPT_STAGING_DIR", "/mnt/data/users/wfz/uv3-checkpoint-staging"
        )) / target.parent.name
        staging_root.mkdir(parents=True, exist_ok=True)
        local_stage = staging_root / f"{target.name}.staged"
        remote_stage = target.with_name(f".{target.name}.staged")
        # Never overwrite the last good OSS checkpoint while serialization is
        # in progress. A SIGBUS/OSS write failure leaves either the old target
        # or a complete local recovery copy, not a truncated ckpt.pt.
        torch.save(payload, local_stage)
        staged = torch.load(local_stage, map_location="cpu", mmap=True, weights_only=False)
        if int(staged.get("step", -1)) != int(payload["step"]):
            raise RuntimeError(f"local checkpoint validation failed: {local_stage}")
        shutil.copyfile(local_stage, remote_stage)
        if remote_stage.stat().st_size != local_stage.stat().st_size:
            raise RuntimeError(f"OSS checkpoint size mismatch: {remote_stage}")
        remote = torch.load(remote_stage, map_location="cpu", mmap=True, weights_only=False)
        if int(remote.get("step", -1)) != int(payload["step"]):
            raise RuntimeError(f"OSS checkpoint validation failed: {remote_stage}")
        os.replace(remote_stage, target)
        local_stage.unlink()
    if dist_ok:
        torch.distributed.barrier()


def load_ckpt(model, optimizers: dict, path: str, extra_models=None, return_payload=False):
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
    )
    dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank0 = (not dist_ok) or torch.distributed.get_rank() == 0
    ckpt = (
        torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if rank0 else {}
    )
    opts = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=dist_ok,
    )
    set_model_state_dict(model, ckpt.get("model", {}), options=opts)
    saved_extra_models = ckpt.get("extra_models", {}) if rank0 else {}
    for name, extra_model in (extra_models or {}).items():
        if rank0 and name not in saved_extra_models:
            raise KeyError(f"checkpoint is missing required extra model: {name}")
        set_model_state_dict(extra_model, saved_extra_models.get(name, {}), options=opts)
    for k, o in optimizers.items():
        optimizer_state = ckpt.get("optim", {}).get(k, {}) if rank0 else {}
        # Adam has no entry for trainable parameters that never received a
        # gradient (for example dormant double-stream modules in a pure-single
        # model). Their empty state is valid and will initialize on first use.
        optimizer_opts = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=dist_ok,
            strict=False,
        )
        set_optimizer_state_dict(model, o, optimizer_state, options=optimizer_opts)
    metadata = {
        "step": int(ckpt.get("step", 0)) if rank0 else 0,
        "rng": ckpt.get("rng") if rank0 else None,
        "rng_by_rank": ckpt.get("rng_by_rank") if rank0 else None,
        "data_status": ckpt.get("data_status") if rank0 else None,
    }
    if dist_ok:
        objects = [metadata if rank0 else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        metadata = objects[0]
    return (metadata["step"], metadata) if return_payload else metadata["step"]
