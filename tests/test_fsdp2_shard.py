"""Muon x FSDP2 correctness: 2-GPU fully_shard+native Muon vs single-GPU Muon.
Ported from review's /tmp/muon_fsdp2_test.py (the only validated comparison).

Single-GPU arm: just verifies Muon runs + update is semi-orthogonal (NS property).
2-GPU arm: same model/grad/seed, compare FSDP2-sharded Muon params vs single-GPU full Muon
  -> cosine ~1, rel diff ~1e-2 (bf16 reduce-order tolerance).

Run 2-GPU: CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node=2 tests/test_fsdp2_shard.py
Single:    CUDA_VISIBLE_DEVICES=5 python tests/test_fsdp2_shard.py
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def _make(seed=0):
    torch.manual_seed(seed)
    # three 2D linears of differing shape (exercises NS aspect-ratio scaling)
    return nn.Sequential(nn.Linear(512, 768, bias=False), nn.ReLU(),
                         nn.Linear(768, 768, bias=False), nn.ReLU(),
                         nn.Linear(768, 256, bias=False))


def _single_step(model, lr=1e-3, seed=42):
    """Single-GPU Muon step with deterministic grads; return post-step params (full tensors)."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt = torch.optim.Muon(list(model.parameters()), lr=lr, momentum=0.95, nesterov=True, ns_steps=5)
    opt.step()
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def main():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        world = torch.distributed.get_world_size()
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

        torch.manual_seed(0)
        m = _make().to(dev)
        # shard each linear leaf then root
        for mod in [m[0], m[2], m[4]]:
            fully_shard(mod, mesh=mesh, mp_policy=mp)
        fully_shard(m, mesh=mesh, mp_policy=mp)
        # deterministic grads (same seed on all ranks -> same gradient values)
        torch.manual_seed(42)
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        opt = torch.optim.Muon(list(m.parameters()), lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5)
        opt.step()
        # gather full params and compare to single-GPU reference
        ref = _single_step(_make(), lr=1e-3, seed=42)
        for n, p in m.named_parameters():
            full = p.full_tensor() if hasattr(p, "full_tensor") else p
            r = ref[n].to(dev)
            cos = torch.nn.functional.cosine_similarity(full.flatten().float(), r.flatten().float(), dim=0).item()
            rel = (full - r).abs().max().item() / (r.abs().max().item() + 1e-9)
            if rank == 0:
                print(f"  {n:20s} cos={cos:.6f} rel={rel:.4e}", flush=True)
                assert cos > 0.999, f"{n}: cosine {cos} < 0.999 (Muon x FSDP2 mismatch)"
        if rank == 0:
            print("test_fsdp2_shard (2-GPU): Muon x FSDP2 == single-GPU Muon (cos>0.999) PASS", flush=True)
    else:
        # single-GPU arm: Muon runs + update semi-orthogonal (NS property on the UPDATE, not weight)
        dev = torch.device("cuda:0")
        m = _make().to(dev)
        W0 = m[0].weight.detach().clone()
        torch.manual_seed(42)
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        opt = torch.optim.Muon(list(m.parameters()), lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5)
        opt.step()
        U = (W0 - m[0].weight.detach()) / 1e-3  # the orthogonalized update
        S = torch.linalg.svdvals(U.float())
        assert (S > 0.4).all() and (S < 1.6).all(), f"update sv out of [0.4,1.6]: {S.tolist()}"
        print(f"test_fsdp2_shard (single-GPU): Muon step OK, update sv in [0.4,1.6] (NS property) PASS", flush=True)


if __name__ == "__main__":
    main()
