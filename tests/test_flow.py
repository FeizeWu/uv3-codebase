"""flow.py unit tests: velocity sign = noise - clean, interpolate endpoints, euler recovers clean."""
import torch
from uv3.modeling.flow import interpolate, velocity_target, euler_schedule, euler_step


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


if __name__ == "__main__":
    test_velocity_sign()
    test_interpolate_endpoints()
    test_euler_recovers_clean()
    print("test_flow: ALL PASS")
