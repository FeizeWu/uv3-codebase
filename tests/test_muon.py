"""Muon optimizer unit tests (v3-corrected).

The Newton-Schulz orthogonalizes the UPDATE (momentum post-NS), NOT the weight.
After one step, W_new = W_old - lr*update; the update U=(W_old-W_new)/lr should have
singular values ~Uniform(0.5,1.5) (the NS property), i.e. approximately semi-orthogonal.
Assert that — NOT W@W.T (the weight is not orthogonal).
"""
import torch
import torch.nn as nn

from uv3.optim.build_optimizers import _is_muon_param


def _ns_update_svd(m: int, n: int, dev="cuda"):
    torch.manual_seed(0)
    lin = nn.Linear(m, n, bias=False).to(dev)
    W0 = lin.weight.detach().clone()
    lin.weight.grad = torch.randn_like(lin.weight)
    lr = 0.01
    opt = torch.optim.Muon([lin.weight], lr=lr, momentum=0.95, nesterov=True, ns_steps=5)
    opt.step()
    U = (W0 - lin.weight.detach()) / lr  # the orthogonalized update
    S = torch.linalg.svdvals(U.float())
    return S


def test_ns_update_semiorthogonal_square():
    S = _ns_update_svd(64, 64)
    # NS produces S' ~ Uniform(0.5, 1.5) (muon.py docstring) -> spread is EXPECTED, not a bug.
    # Assert range only (the real NS property); the spread (up to ~1.0) is by design.
    assert (S > 0.4).all() and (S < 1.6).all(), f"sv out of range: {S.tolist()}"
    assert S.mean().item() > 0.7 and S.mean().item() < 1.3, f"mean sv off: {S.mean().item()}"


def test_ns_update_semiorthogonal_tall():
    S = _ns_update_svd(128, 64)
    assert (S > 0.4).all() and (S < 1.6).all(), f"tall sv out of range: {S.tolist()}"


def test_muon_rejects_1d():
    lin = nn.Linear(8, 8)
    try:
        torch.optim.Muon([lin.bias], lr=0.01)
        raise AssertionError("Muon should reject 1D params")
    except (ValueError, RuntimeError):
        pass


def test_param_split_predicate():
    from diffusers import Flux2Transformer2DModel
    m = Flux2Transformer2DModel(
        in_channels=128, out_channels=128, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, guidance_embeds=False,
    )
    names = dict(m.named_parameters())
    # a 2D weight inside transformer_blocks -> Muon
    assert _is_muon_param("transformer_blocks.0.attn.to_q.weight", names["transformer_blocks.0.attn.to_q.weight"])
    # context_embedder (2D but not in blocks) -> Adam
    assert not _is_muon_param("context_embedder.weight", names["context_embedder.weight"])
    # a bias (1D) -> Adam
    import torch
    bias_name = [n for n, _ in m.named_parameters() if _.ndim == 1][0]
    assert not _is_muon_param(bias_name, names[bias_name])


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_ns_update_semiorthogonal_square()
    test_ns_update_semiorthogonal_tall()
    test_muon_rejects_1d()
    test_param_split_predicate()
    print("test_muon: ALL PASS")
