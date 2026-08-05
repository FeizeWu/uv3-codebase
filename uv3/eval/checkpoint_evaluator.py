"""Distributed fixed-case generation and FID/KID evaluation for UV3 checkpoints."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from ..config import load_config
from ..data.tar_dataset import TarMetadataDataset
from ..data.transforms import center_crop_resize
from ..modeling.flow import euler_schedule, euler_step
from ..train.fsdp2 import apply_fsdp2, make_mesh
from ..train.fsdp2_trainer import _attention_mask, build


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    return images[:, :3].float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    array = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_reference(case: dict[str, Any], image_size: int) -> Image.Image:
    with open(case["image_tar"], "rb") as file:
        file.seek(case["offset"])
        raw = file.read(case["size"])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    # Use the exact training/FID transform so side-by-side visual comparisons
    # do not mistake letterboxing or a different crop for a model error.
    return center_crop_resize(image, image_size)


def load_eval_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("evaluation", raw)


def load_checkpoint_weights(model, checkpoint: Path, use_ema: bool) -> int:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    rank = torch.distributed.get_rank()
    state = {}
    step = 0
    if rank == 0:
        # Only rank 0 reads the monolithic checkpoint from OSS. FSDP broadcasts
        # parameters while loading; eight independent reads multiplied a 29GB
        # checkpoint into ~232GB of storage traffic.
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        if use_ema:
            state = checkpoint_data.get("extra_models", {}).get("self_flow_teacher")
            if state is None:
                raise KeyError("Self-Flow checkpoint has no self_flow_teacher EMA state")
        else:
            state = checkpoint_data["model"]
            # Self-Flow training saves ModuleList([mmdit, projector]); extract MMDiT.
            if state and all(key.startswith("0.") or key.startswith("1.") for key in state):
                state = {key[2:]: value for key, value in state.items() if key.startswith("0.")}
        step = int(checkpoint_data.get("step", 0))
    set_model_state_dict(
        model,
        state,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,
        ),
    )
    step_tensor = torch.tensor(step, device=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    torch.distributed.broadcast(step_tensor, src=0)
    return int(step_tensor.item())


@torch.inference_mode()
def encode_prompts(qwen, model, prompts: list[str], image_tokens: int, cfg, device):
    ids, valid = qwen.tokenize(prompts, device, max_length=cfg.model.qwen_vl.max_length)
    text = qwen.encode_text(ids, valid)
    mask = _attention_mask(model, valid, image_tokens, cfg.train.block_size, device)
    return text, mask


@torch.inference_mode()
def sample_batch(model, vae, text, attention_mask, seeds, steps: int, image_size: int):
    device, dtype = text.device, text.dtype
    channels, spatial = vae.latent_spec(image_size)
    noises = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noises.append(
            torch.randn((channels, spatial, spatial), device=device, dtype=dtype, generator=generator)
        )
    latents = torch.stack(noises)
    times = euler_schedule(steps, device, dtype)
    for current, following in zip(times[:-1], times[1:]):
        velocity = model.predict_velocity(
            latents, text, current.expand(latents.shape[0]), text_attn_mask=attention_mask
        )
        latents = euler_step(latents, velocity, current, following)
    return tensor_to_uint8(vae.decode_latents(latents.to(vae.dtype)))


def fixed_case_evaluation(model, vae, qwen, cfg, eval_cfg, output: Path, rank: int, world: int, max_cases: int | None):
    all_cases = read_jsonl(Path(eval_cfg["fixed_cases_manifest"]))
    if max_cases is not None:
        all_cases = all_cases[:max_cases]
    # Equal work and tensor shapes on all FSDP ranks. Padding repeats are not saved.
    per_rank = math.ceil(len(all_cases) / world)
    padded = all_cases + [all_cases[-1]] * (per_rank * world - len(all_cases))
    local = padded[rank::world]
    prompts = [case["caption"] for case in local]
    image_size = int(eval_cfg.get("image_size", cfg.data.image_size))
    image_tokens = (image_size // 16) ** 2
    text, attention_mask = encode_prompts(
        qwen, model, prompts, image_tokens, cfg, torch.device("cuda", rank)
    )
    generated = sample_batch(
        model, vae, text, attention_mask, [case["seed"] for case in local],
        int(eval_cfg.get("sample_steps", 30)), image_size,
    )
    samples_dir = output / "fixed"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for local_index, (case, image) in enumerate(zip(local, generated)):
        global_index = rank + local_index * world
        if global_index >= len(all_cases):
            continue
        generated_name = f'{global_index:04d}_{case["split"]}_generated.png'
        reference_name = f'{global_index:04d}_{case["split"]}_reference.png'
        save_tensor_image(image, samples_dir / generated_name)
        load_reference(case, image_size).save(samples_dir / reference_name)
        rows.append(
            {
                **case,
                "generated": f"fixed/{generated_name}",
                "reference": f"fixed/{reference_name}",
                "sample_steps": int(eval_cfg.get("sample_steps", 30)),
            }
        )
    rank_manifest = output / f"manifest.rank{rank}.jsonl"
    with rank_manifest.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.distributed.barrier()
    if rank == 0:
        merged = []
        for current_rank in range(world):
            merged.extend(read_jsonl(output / f"manifest.rank{current_rank}.jsonl"))
        merged.sort(key=lambda row: row["case_id"])
        with (output / "manifest.jsonl").open("w", encoding="utf-8") as file:
            for row in merged:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.distributed.barrier()


def distribution_evaluation(model, vae, qwen, cfg, eval_cfg, rank: int, world: int, max_generated: int | None):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    metric_cfg = eval_cfg.get("distribution_metrics", {})
    total = int(max_generated or metric_cfg.get("num_generated", 10_000))
    if total % world:
        raise ValueError(f"num_generated={total} must be divisible by world_size={world}")
    local_target = total // world
    batch_size = int(metric_cfg.get("batch_size_per_gpu", 1))
    image_size = int(eval_cfg.get("image_size", cfg.data.image_size))
    dataset = TarMetadataDataset(
        eval_cfg["holdout_manifest"], image_size, cfg.data.caption_field, shuffle=False
    )
    iterator = iter(dataset)
    metric_device = torch.device("cuda", rank)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    kid = KernelInceptionDistance(subset_size=min(1000, total)).to(metric_device)
    clip_model = clip_processor = None
    clip_path = metric_cfg.get("clip_model_path")
    if clip_path:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained(
            clip_path, local_files_only=True
        ).to(metric_device).eval()
        clip_processor = CLIPProcessor.from_pretrained(
            clip_path, local_files_only=True
        )
    clip_sum = torch.zeros((), device=metric_device, dtype=torch.float64)
    clip_count = torch.zeros((), device=metric_device, dtype=torch.float64)
    processed = 0
    while processed < local_target:
        current = min(batch_size, local_target - processed)
        rows = [next(iterator) for _ in range(current)]
        real = tensor_to_uint8(torch.stack([row["pixel_values"] for row in rows]).to(rank))
        prompts = [row["text"] for row in rows]
        image_tokens = (image_size // 16) ** 2
        text, attention_mask = encode_prompts(qwen, model, prompts, image_tokens, cfg, torch.device("cuda", rank))
        seeds = [20260805 + rank * local_target + processed + index for index in range(current)]
        fake = sample_batch(
            model, vae, text, attention_mask, seeds,
            int(eval_cfg.get("sample_steps", 30)), image_size,
        )
        fid.update(real, real=True); fid.update(fake, real=False)
        kid.update(real, real=True); kid.update(fake, real=False)
        if clip_model is not None:
            pil_images = [
                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()) for image in fake
            ]
            clip_inputs = clip_processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(metric_device)
            clip_output = clip_model(**clip_inputs)
            image_features = torch.nn.functional.normalize(
                clip_output.image_embeds.float(), dim=-1
            )
            text_features = torch.nn.functional.normalize(
                clip_output.text_embeds.float(), dim=-1
            )
            scores = 100.0 * (image_features * text_features).sum(dim=-1).clamp_min(0)
            clip_sum += scores.double().sum()
            clip_count += scores.numel()
        processed += current
        if rank == 0 and processed % max(batch_size * 10, 1) == 0:
            print(f"[eval] distribution local={processed}/{local_target}", flush=True)
    fid_value = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    if clip_model is not None:
        torch.distributed.all_reduce(clip_sum)
        torch.distributed.all_reduce(clip_count)
        clipscore = float((clip_sum / clip_count.clamp_min(1)).item())
        clipscore_note = None
    else:
        clipscore = None
        clipscore_note = "not computed: clip_model_path is not configured"
    return {
        "num_generated": total,
        "fid": fid_value,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "clipscore": clipscore,
        "clipscore_note": clipscore_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-fixed", type=int)
    parser.add_argument("--max-generated", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--student", action="store_true", help="evaluate student instead of EMA")
    args = parser.parse_args()
    if args.checkpoint is None and not args.allow_random_init:
        parser.error("--checkpoint is required unless --allow-random-init is set")

    torch.distributed.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cfg = load_config(str(args.train_config))
    eval_cfg = load_eval_config(args.eval_config)
    configured_weights = str(eval_cfg.get("weights", "ema_teacher"))
    if configured_weights not in {"student", "ema_teacher"}:
        raise ValueError(f"evaluation.weights must be student or ema_teacher, got {configured_weights!r}")
    use_student = args.student or configured_weights == "student"
    if args.sample_steps is not None:
        eval_cfg["sample_steps"] = args.sample_steps
    # Evaluation uses one MMDiT only; no projector or duplicate teacher model.
    vae, qwen, model = build(cfg, device, torch.bfloat16)
    mesh = make_mesh(num_replicate=cfg.train.num_replicate, num_shard=cfg.train.num_shard)
    apply_fsdp2(model, mesh, reshard_after_forward=True)
    step = 0
    if args.checkpoint is not None:
        step = load_checkpoint_weights(model, args.checkpoint, use_ema=not use_student)
    model.eval()
    output = args.run_dir / "samples" / f"step_{step:08d}"
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_fixed:
        fixed_case_evaluation(
            model, vae, qwen, cfg, eval_cfg, output, rank, world, args.max_fixed
        )
    results = None
    distribution_enabled = bool(
        eval_cfg.get("distribution_metrics", {}).get("enabled", True)
    )
    if not args.skip_distribution and distribution_enabled:
        results = distribution_evaluation(
            model, vae, qwen, cfg, eval_cfg, rank, world, args.max_generated
        )
    if rank == 0:
        metadata = {
            "step": step,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "weights": "student" if use_student else "ema_teacher",
            "random_init_smoke": args.checkpoint is None,
            "evaluation": eval_cfg,
            "metrics": results,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[eval] DONE output={output}", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
