from __future__ import annotations

import random

import numpy as np
import torch

from uv3.train.fsdp2 import load_ckpt, save_ckpt


def _adam_step(model, optimizer):
    optimizer.zero_grad()
    model(torch.ones(2, 4)).float().sum().backward()
    optimizer.step()


def test_checkpoints_are_retained_and_latest_pointer_advances(tmp_path, monkeypatch):
    monkeypatch.setenv("UV3_CKPT_STAGING_DIR", str(tmp_path / "staging"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    latest = run_dir / "ckpt.pt"
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _adam_step(model, optimizer)
    model._step = 10
    save_ckpt(model, {"adam": optimizer}, str(latest))
    _adam_step(model, optimizer)
    model._step = 20
    save_ckpt(model, {"adam": optimizer}, str(latest))

    first = run_dir / "ckpt_step_00000010.pt"
    second = run_dir / "ckpt_step_00000020.pt"
    assert first.is_file()
    assert second.is_file()
    assert latest.is_file()
    assert int(torch.load(first, weights_only=False)["step"]) == 10
    assert int(torch.load(second, weights_only=False)["step"]) == 20
    assert int(torch.load(latest, weights_only=False)["step"]) == 20


def test_rng_checkpoint_contains_torch_python_and_numpy(tmp_path, monkeypatch):
    monkeypatch.setenv("UV3_CKPT_STAGING_DIR", str(tmp_path / "staging"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model._step = 1
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    save_ckpt(model, {"adam": optimizer}, str(run_dir / "ckpt.pt"))
    state = torch.load(run_dir / "ckpt.pt", weights_only=False)["rng"]
    assert {"torch", "python", "numpy"}.issubset(state)


def test_self_flow_checkpoint_can_resume_into_student_only(tmp_path, monkeypatch):
    monkeypatch.setenv("UV3_CKPT_STAGING_DIR", str(tmp_path / "staging"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    student = torch.nn.Linear(4, 3)
    projector = torch.nn.Linear(3, 3)
    wrapped = torch.nn.ModuleList([student, projector])
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    optimizer.zero_grad()
    projector(student(torch.ones(2, 4))).sum().backward()
    optimizer.step()
    wrapped._step = 17
    expected = {name: value.detach().clone() for name, value in student.state_dict().items()}
    save_ckpt(wrapped, {"adam": optimizer}, str(run_dir / "ckpt.pt"))

    resumed = torch.nn.Linear(4, 3)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    step = load_ckpt(
        resumed,
        {"adam": resumed_optimizer},
        str(run_dir / "ckpt.pt"),
        allow_self_flow_disable=True,
    )

    assert step == 17
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert resumed_optimizer.state
    assert all(
        state["exp_avg"].dtype == torch.float32
        and state["exp_avg_sq"].dtype == torch.float32
        for state in resumed_optimizer.state.values()
    )
