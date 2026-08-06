"""FSDP2 native wrapper (fully_shard). Pattern from transfusion-core fsdp2_utils_mot.

Shards MMDiT transformer blocks (leaf) then root. Native torch.optim.Muon works under
FSDP2 (review's 2-GPU test: cos~0.99997 via DTensor implicit collectives; NOT officially
supported but empirically correct on torch 2.12). ckpt via checkpoint.state_dict.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
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
    if hasattr(model, "compute_dtype"):
        # Outside FSDP the FP32 master parameters also compute in FP32. Once
        # wrapped, the policy casts parameters/inputs for BF16 compute while
        # leaving the persistent parameter storage in FP32.
        model.compute_dtype = param_dtype
    return model


def _step_checkpoint_path(target: Path, step: int) -> Path:
    return target.with_name(f"{target.stem}_step_{int(step):08d}{target.suffix}")


def _validated_checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    step = int(payload.get("step", -1))
    if step < 0:
        raise RuntimeError(f"checkpoint has invalid step: {path}")
    return step


def _publish_latest_pointer(target: Path, retained: Path) -> None:
    """Atomically point ckpt.pt at an immutable step file, with a copy fallback."""
    temporary = target.with_name(f".{target.name}.latest")
    if os.path.lexists(temporary):
        temporary.unlink()
    try:
        temporary.symlink_to(retained.name)
        os.replace(temporary, target)
    except OSError:
        if os.path.lexists(temporary):
            temporary.unlink()
        copy_stage = target.with_name(f".{target.name}.staged")
        shutil.copyfile(retained, copy_stage)
        if copy_stage.stat().st_size != retained.stat().st_size:
            raise RuntimeError(f"latest checkpoint copy size mismatch: {copy_stage}")
        os.replace(copy_stage, target)


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
    rng_state = {
        # Keep the legacy key for old readers while naming all RNGs explicitly.
        "py": torch.get_rng_state(),
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
    if rng is not None and hasattr(rng, "get_state"):
        rng_state["py_gen"] = rng.get_state()
    dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank0 = (not dist_ok) or torch.distributed.get_rank() == 0
    if dist_ok:
        rng_by_rank = [None] * torch.distributed.get_world_size() if rank0 else None
        torch.distributed.gather_object(rng_state, rng_by_rank, dst=0)
        data_status_by_rank = [None] * torch.distributed.get_world_size() if rank0 else None
        torch.distributed.gather_object(data_status, data_status_by_rank, dst=0)
    else:
        rng_by_rank = [rng_state]
        data_status_by_rank = [data_status]
    if rank0:
        payload = {"model": msd, "optim": osd, "step": getattr(model, "_step", 0),
                   "rng": rng_state, "rng_by_rank": rng_by_rank,
                   "data_status": data_status,
                   "data_status_by_rank": data_status_by_rank,
                   "extra_models": extra_model_states}
        target = Path(path)
        step = int(payload["step"])
        retained = _step_checkpoint_path(target, step)
        staging_root = Path(os.environ.get(
            "UV3_CKPT_STAGING_DIR", "/mnt/data/users/wfz/uv3-checkpoint-staging"
        )) / target.parent.name
        staging_root.mkdir(parents=True, exist_ok=True)
        local_stage = staging_root / f"{retained.name}.staged"
        remote_stage = retained.with_name(f".{retained.name}.staged")
        # Never overwrite the last good OSS checkpoint while serialization is
        # in progress. A SIGBUS/OSS write failure leaves either the old target
        # or a complete local recovery copy, not a truncated ckpt.pt.
        # Preserve a pre-upgrade ckpt.pt before replacing it with a latest pointer.
        if target.exists() and not target.is_symlink():
            previous_step = _validated_checkpoint_step(target)
            previous_retained = _step_checkpoint_path(target, previous_step)
            if not previous_retained.exists():
                shutil.copyfile(target, previous_retained)
                if _validated_checkpoint_step(previous_retained) != previous_step:
                    raise RuntimeError(
                        f"failed to retain previous checkpoint: {previous_retained}"
                    )
        if retained.exists() and _validated_checkpoint_step(retained) == step:
            _publish_latest_pointer(target, retained)
        else:
            torch.save(payload, local_stage)
            if _validated_checkpoint_step(local_stage) != step:
                raise RuntimeError(f"local checkpoint validation failed: {local_stage}")
            shutil.copyfile(local_stage, remote_stage)
            if remote_stage.stat().st_size != local_stage.stat().st_size:
                raise RuntimeError(f"OSS checkpoint size mismatch: {remote_stage}")
            if _validated_checkpoint_step(remote_stage) != step:
                raise RuntimeError(f"OSS checkpoint validation failed: {remote_stage}")
            os.replace(remote_stage, retained)
            _publish_latest_pointer(target, retained)
            local_stage.unlink()
    if dist_ok:
        torch.distributed.barrier()


def _unwrap_self_flow_model_state(model_state: dict) -> dict:
    student = {key[2:]: value for key, value in model_state.items() if key.startswith("0.")}
    projector = [key for key in model_state if key.startswith("1.")]
    unexpected = [key for key in model_state if not key.startswith(("0.", "1."))]
    if not student or not projector or unexpected:
        raise RuntimeError(
            "checkpoint is not a recognized Self-Flow ModuleList layout: "
            f"student={len(student)} projector={len(projector)} unexpected={unexpected[:3]}"
        )
    return student


def _unwrap_self_flow_optimizer_state(optimizer_state: dict) -> dict:
    state = optimizer_state.get("state", {})
    if state and not all(isinstance(key, str) for key in state):
        raise RuntimeError("Self-Flow optimizer conversion requires FQN-keyed state")
    converted = {
        key: value
        for key, value in optimizer_state.items()
        if key not in {"state", "param_groups"}
    }
    converted["state"] = {
        key[2:]: value for key, value in state.items() if key.startswith("0.")
    }
    converted_groups = []
    for original_group in optimizer_state.get("param_groups", []):
        group = dict(original_group)
        params = group.get("params", [])
        if not all(isinstance(key, str) for key in params):
            raise RuntimeError("Self-Flow optimizer conversion requires FQN param groups")
        group["params"] = [key[2:] for key in params if key.startswith("0.")]
        if group["params"]:
            converted_groups.append(group)
    converted["param_groups"] = converted_groups
    return converted


def load_ckpt(
    model,
    optimizers: dict,
    path: str,
    extra_models=None,
    return_payload=False,
    allow_self_flow_disable: bool = False,
):
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
    model_state = ckpt.get("model", {})
    if allow_self_flow_disable and any(key.startswith("1.") for key in model_state):
        model_state = _unwrap_self_flow_model_state(model_state)
    set_model_state_dict(model, model_state, options=opts)
    saved_extra_models = ckpt.get("extra_models", {}) if rank0 else {}
    for name, extra_model in (extra_models or {}).items():
        if rank0 and name not in saved_extra_models:
            raise KeyError(f"checkpoint is missing required extra model: {name}")
        set_model_state_dict(extra_model, saved_extra_models.get(name, {}), options=opts)
    for k, o in optimizers.items():
        optimizer_state = ckpt.get("optim", {}).get(k, {}) if rank0 else {}
        if allow_self_flow_disable and rank0 and optimizer_state:
            optimizer_state = _unwrap_self_flow_optimizer_state(optimizer_state)
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
        "data_status_by_rank": ckpt.get("data_status_by_rank") if rank0 else None,
    }
    if dist_ok:
        objects = [metadata if rank0 else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        metadata = objects[0]
    return (metadata["step"], metadata) if return_payload else metadata["step"]
