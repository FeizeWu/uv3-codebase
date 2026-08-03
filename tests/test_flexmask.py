"""test_flexmask: flex document BlockMask == dense SDPA (padded q/k/v, bf16 tol ~1e-2)."""
import torch
from uv3.train.flex_attn import (
    build_document_block_mask,
    build_padding_block_mask,
    dense_document_mask,
    flex_attn,
)


def _pad(x, P):
    return torch.nn.functional.pad(x, (0, 0, 0, P - x.shape[2]))


def test_document_mask_matches_sdpa_single():
    """single sample: flex (padding+full) == SDPA (padding+full) on valid positions."""
    torch.manual_seed(0)
    dev = "cuda"
    H, D, L = 4, 64, 12
    P = 128
    q = torch.randn(1, H, L, D, device=dev, dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q)
    qp, kp, vp = _pad(q, P), _pad(k, P), _pad(v, P)

    bm = build_document_block_mask([L], H, dev, block_size=P, _compile=False)
    out_flex = flex_attn(qp, kp, vp, bm)            # (1,H,P,D)

    dense = torch.full((1, 1, P, P), float("-inf"), device=dev)
    dense[0, 0, :L, :L] = 0.0
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, attn_mask=dense)

    diff = (out_flex[0, :, :L] - out_sdpa[0, :, :L]).abs().max()
    assert diff < 0.05, f"flex vs sdpa max diff {diff}"


def test_document_mask_matches_sdpa_multi():
    """two packed samples: no cross-sample attention; each attends fully to itself."""
    torch.manual_seed(1)
    dev = "cuda"
    H, D = 4, 64
    seq_lens = [12, 8]
    L = max(seq_lens); P = 128
    q = torch.randn(2, H, L, D, device=dev, dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q)
    qp, kp, vp = _pad(q, P), _pad(k, P), _pad(v, P)

    bm = build_document_block_mask(seq_lens, H, dev, block_size=P, _compile=False)
    out_flex = flex_attn(qp, kp, vp, bm)

    # dense: sample i attends its own [0:n_i, 0:n_i], padding masked
    dense = torch.full((2, 1, P, P), float("-inf"), device=dev)
    for i, n in enumerate(seq_lens):
        dense[i, 0, :n, :n] = 0.0
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, attn_mask=dense)

    for i, n in enumerate(seq_lens):
        diff = (out_flex[i, :, :n] - out_sdpa[i, :, :n]).abs().max()
        assert diff < 0.05, f"sample {i} flex vs sdpa diff {diff}"


def test_padding_hole_keeps_image_tokens_valid():
    """Static [text, image] layout masks text padding without masking later image tokens."""
    torch.manual_seed(2)
    dev = "cuda"
    H, D, L, P = 4, 64, 12, 128
    valid = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device=dev).bool()
    q = torch.randn(1, H, L, D, device=dev)
    k = torch.randn_like(q); v = torch.randn_like(q)
    qp, kp, vp = _pad(q, P), _pad(k, P), _pad(v, P)

    bm = build_padding_block_mask(valid, H, block_size=P)
    out_flex = flex_attn(qp, kp, vp, bm)
    dense = torch.full((1, 1, P, P), float("-inf"), device=dev)
    idx = valid[0].nonzero().flatten()
    dense[0, 0, idx[:, None], idx[None, :]] = 0.0
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, attn_mask=dense)
    diff = (out_flex[0, :, idx] - out_sdpa[0, :, idx]).abs().max()
    assert diff < 0.05, f"padding-hole flex vs sdpa diff {diff}"


if __name__ == "__main__":
    test_document_mask_matches_sdpa_single()
    test_document_mask_matches_sdpa_multi()
    test_padding_hole_keeps_image_tokens_valid()
    print("test_flexmask: ALL PASS")
