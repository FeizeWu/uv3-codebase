"""Small runtime smoke for UV3's node-local-shard HSDP mesh."""
from __future__ import annotations

import os

import torch
from torch.distributed._composable.fsdp import fully_shard

from uv3.train.fsdp2 import load_ckpt, make_mesh, save_ckpt


def main() -> None:
    torch.distributed.init_process_group("nccl")
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Four local ranks emulate two nodes with two-way node-local sharding.
    mesh = make_mesh(num_shard=2)
    assert tuple(mesh.shape) == (2, 2)
    torch.manual_seed(123)
    model = torch.nn.Linear(32, 32, bias=False).to(device)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Different inputs on every DP rank verify that HSDP reduces gradients over
    # both the shard and replicate dimensions and leaves one consistent model.
    generator = torch.Generator(device=device).manual_seed(1000 + global_rank)
    inputs = torch.randn(8, 32, device=device, generator=generator)
    model(inputs).square().mean().backward()
    optimizer.step()

    weight = model.weight.full_tensor().detach()
    gathered = [torch.empty_like(weight) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, weight)
    for other in gathered:
        torch.testing.assert_close(weight, other)

    # Checkpoint metadata must preserve a different RNG stream for each global
    # rank; broadcasting rank 0's RNG to all ranks would correlate resumed
    # noise/timestep samples.
    torch.manual_seed(2000 + global_rank)
    own_rng = torch.get_rng_state().clone()
    checkpoint = "/tmp/uv3_hsdp_runtime_ckpt.pt"
    model._step = 7
    save_ckpt(model, {"adam": optimizer}, checkpoint)
    step, metadata = load_ckpt(
        model, {"adam": optimizer}, checkpoint, return_payload=True
    )
    assert step == 7
    assert len(metadata["rng_by_rank"]) == torch.distributed.get_world_size()
    torch.testing.assert_close(metadata["rng_by_rank"][global_rank]["py"], own_rng)
    assert not torch.equal(
        metadata["rng_by_rank"][0]["py"], metadata["rng_by_rank"][1]["py"]
    )
    if global_rank == 0:
        os.unlink(checkpoint)
        print(
            "HSDP runtime PASS mesh=(2, 2), parameters identical, "
            "per-rank RNG checkpointed"
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
