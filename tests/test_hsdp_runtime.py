"""Small runtime smoke for UV3's node-local-shard HSDP mesh."""
from __future__ import annotations

import copy
import os

import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

from uv3.train.fsdp2 import load_ckpt, make_mesh, save_ckpt
from uv3.train.fsdp2_trainer import _ema_update_local_shards_


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
    teacher = copy.deepcopy(model)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        # Match production: student parameters stay materialized between
        # forward and backward, while the frozen teacher reshards immediately.
        reshard_after_forward=False,
    )
    fully_shard(
        teacher,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        reshard_after_forward=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Different inputs on every DP rank verify that HSDP reduces gradients over
    # both the shard and replicate dimensions and leaves one consistent model.
    generator = torch.Generator(device=device).manual_seed(1000 + global_rank)
    inputs = torch.randn(8, 32, device=device, generator=generator)
    for _ in range(8):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        _ema_update_local_shards_(teacher, model, decay=0.9999)
    assert model.weight.dtype == torch.float32
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

    weight = model.weight.full_tensor().detach()
    gathered = [torch.empty_like(weight) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, weight)
    for other in gathered:
        torch.testing.assert_close(weight, other)

    # The production EMA helper must update matching local FP32 shards without
    # a full-parameter all-gather on every parameter. Compare the resulting
    # full tensor with an independently computed FP32 reference.
    teacher_before = teacher.weight.full_tensor().detach().clone()
    expected_teacher = teacher_before.lerp(weight, 0.25)
    _ema_update_local_shards_(teacher, model, decay=0.75)
    teacher_after = teacher.weight.full_tensor().detach()
    torch.testing.assert_close(teacher_after, expected_teacher)

    # Checkpoint metadata must preserve a different RNG stream for each global
    # rank; broadcasting rank 0's RNG to all ranks would correlate resumed
    # noise/timestep samples.
    torch.manual_seed(2000 + global_rank)
    own_rng = torch.get_rng_state().clone()
    checkpoint = "/tmp/uv3_hsdp_runtime_ckpt.pt"
    model._step = 7
    save_ckpt(
        model,
        {"adam": optimizer},
        checkpoint,
        data_status={"rank": global_rank, "cursor": global_rank * 10 + 3},
    )
    step, metadata = load_ckpt(
        model, {"adam": optimizer}, checkpoint, return_payload=True
    )
    assert step == 7
    assert len(metadata["rng_by_rank"]) == torch.distributed.get_world_size()
    torch.testing.assert_close(metadata["rng_by_rank"][global_rank]["py"], own_rng)
    assert not torch.equal(
        metadata["rng_by_rank"][0]["py"], metadata["rng_by_rank"][1]["py"]
    )
    assert metadata["data_status_by_rank"][global_rank] == {
        "rank": global_rank,
        "cursor": global_rank * 10 + 3,
    }
    if global_rank == 0:
        os.unlink(checkpoint)
        os.unlink("/tmp/uv3_hsdp_runtime_ckpt_step_00000007.pt")
        print(
            "HSDP runtime PASS mesh=(2, 2), BF16 compute with FP32 storage, "
            "parameters/EMA identical, per-rank RNG/data cursor checkpointed"
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
