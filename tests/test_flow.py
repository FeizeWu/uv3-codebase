"""flow.py unit tests: velocity sign = noise - clean, interpolate endpoints, euler recovers clean."""
import torch
from uv3.modeling.flow import interpolate, velocity_target, euler_schedule, euler_step
from uv3.data.noise_scheduler import sample_timesteps, timestep_bin_sums


def test_velocity_sign():
    c = torch.randn(2, 3, 4, 4)
    n = torch.randn_like(c)
    # v = d/dt[(1-t)c + t n] = n - c   (t=0 clean -> t=1 noise)
    assert torch.allclose(velocity_target(c, n), n - c), "velocity must be noise-clean"
    assert not torch.allclose(velocity_target(c, n), c - n), "must NOT be clean-noise"


def test_interpolate_endpoints():
    c = torch.randn(2, 3)
    n = torch.randn_like(c)
    assert torch.allclose(interpolate(c, n, torch.zeros(2)), c, atol=1e-6)
    assert torch.allclose(interpolate(c, n, torch.ones(2)), n, atol=1e-6)


def test_euler_recovers_clean():
    # exact velocity v=n-c, integrate from t=1 (noise) to 0 -> should give clean
    c = torch.randn(2, 8)
    n = torch.randn_like(c)
    x = n.clone()
    times = euler_schedule(1000, c.device, torch.float32)
    for cur, foll in zip(times[:-1], times[1:]):
        x = euler_step(x, n - c, cur, foll)
    assert torch.allclose(x, c, atol=1e-2), f"euler did not recover clean: max_err={(x - c).abs().max()}"


def test_timestep_sampler_direction_and_bins():
    torch.manual_seed(7)
    uniform = sample_timesteps(4096, "cpu", strategy="uniform")
    shifted = sample_timesteps(
        4096, "cpu", strategy="logit_normal_shift", shift=2.0
    )
    assert 0 <= uniform.min() and uniform.max() <= 1
    assert shifted.mean() > 0.5  # shift>1 moves mass toward UV3's noisy endpoint
    t = torch.tensor([0.0, 0.099, 0.1, 0.999, 1.0])
    losses = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    counts, sums, _ = timestep_bin_sums(t, losses)
    assert counts[[0, 1, 9]].tolist() == [2, 1, 2]
    assert sums[[0, 1, 9]].tolist() == [4, 5, 16]


if __name__ == "__main__":
    test_velocity_sign()
    test_interpolate_endpoints()
    test_euler_recovers_clean()
    print("test_flow: ALL PASS")
