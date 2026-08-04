"""Two-GPU smoke test for torchao FP8 FSDP2 weight all-gather.

Run with:
  PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/smoke_fp8_fsdp_all_gather.py
"""
from __future__ import annotations

import os

import torch

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torchao.float8.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp

from uv3.config import ComponentConfig, ModelConfig, SelfFlowConfig
from uv3.modeling.mmdit import MMDiT
from uv3.train.fsdp2 import apply_fsdp2, make_mesh
from uv3.train.fsdp2_trainer import _attention_mask


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16

    cfg = ModelConfig(
        hidden_size=256,
        num_layers=1,
        num_double_layers=1,
        num_single_layers=1,
        num_heads=4,
        latent_channels=32,
        patch_size=2,
        in_channels=128,
        out_channels=128,
        rope_theta=2000.0,
        axes_dims_rope=(16, 16, 16, 16),
        guidance_embeds=False,
        flex_attention=True,
        alpha_on=False,
        self_flow=SelfFlowConfig(enabled=False),
        vae=ComponentConfig(),
        qwen_vl=ComponentConfig(),
        transformer=ComponentConfig(backend="random", trainable=True),
    )
    text_stub = type("TextStub", (), {"hidden_size": 256})
    model = MMDiT.build(cfg.transformer, cfg, text_encoder=text_stub()).to(
        device, dtype=dtype
    )

    def token_block_linear(module, fqn):
        qualified = f".{fqn}."
        return isinstance(module, torch.nn.Linear) and (
            ".transformer_blocks." in qualified
            or ".single_transformer_blocks." in qualified
        )

    convert_to_float8_training(
        model,
        module_filter_fn=token_block_linear,
        config=Float8LinearConfig(enable_fsdp_float8_all_gather=True),
    )
    model.compile(dynamic=False, mode="default")
    smoke_num_shard = os.environ.get("UV3_SMOKE_NUM_SHARD")
    mesh = make_mesh(
        num_shard=int(smoke_num_shard) if smoke_num_shard is not None else None
    )
    apply_fsdp2(model, mesh, reshard_after_forward=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    generator = torch.Generator(device).manual_seed(1234 + rank)
    noisy = torch.randn(8, 32, 16, 16, device=device, dtype=dtype, generator=generator)
    text = torch.randn(8, 8, 256, device=device, dtype=dtype, generator=generator)
    valid = torch.ones(8, 8, device=device, dtype=torch.long)
    mask = _attention_mask(model, valid, n_img=64, block_size=128, device=device)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(noisy, text, text_attn_mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        precompute_float8_dynamic_scale_for_fsdp(model)
        finite = torch.tensor(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            device=device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if rank == 0:
            print(f"step={step} loss={loss.item():.6f} finite={bool(finite.item())}")
        if not finite.item():
            raise RuntimeError("non-finite FP8 FSDP gradient")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
