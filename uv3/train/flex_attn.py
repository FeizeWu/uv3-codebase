"""FlexAttention document mask for the MMDiT (bidirectional + document + padding).

NOT reusing transfusion-core create_sparse_mask (that is MoT-tailored: causal/full/noise
token routing). Here diffusion is BIDIRECTIONAL (no causal; causal only lives in the Qwen3.5
LM head). We need only: each packed sample attends fully to itself, no cross-sample, no pad.
Patterns (build BlockMask outside forward, _compile=False, pad to 128, head_dim>=16) from
diffusers transformer_anyflow_far._build_far_block_mask_from_far_cfg.
"""
from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

try:
    from torch.nn.attention.flex_attention import _compiled_flex_attention  # pre-compiled entry
    _HAS_COMPILED = True
except ImportError:
    _HAS_COMPILED = False


def build_document_block_mask(seq_lens, num_heads, device, block_size: int = 128, _compile: bool = False):
    """BlockMask where token q attends kv iff same sample and neither is padding."""
    max_len = max(seq_lens)
    padded = ((max_len + block_size - 1) // block_size) * block_size
    B = len(seq_lens)
    sample_id = torch.full((B, padded), -1, device=device, dtype=torch.long)
    for i, n in enumerate(seq_lens):
        sample_id[i, :n] = i

    def mask_mod(b, h, q_idx, kv_idx):
        q_ok = sample_id[b, q_idx] >= 0
        kv_ok = sample_id[b, kv_idx] >= 0
        same = sample_id[b, q_idx] == sample_id[b, kv_idx]
        return q_ok & kv_ok & same

    return create_block_mask(
        mask_mod, B=B, H=num_heads, Q_LEN=padded, KV_LEN=padded,
        device=device, BLOCK_SIZE=block_size, _compile=_compile,
    )


def build_padding_block_mask(valid_tokens, num_heads, block_size: int = 128, _compile: bool = False):
    """Build a bidirectional BlockMask from a per-token validity mask.

    ``valid_tokens`` is (B, L) and may contain holes.  That matters for MMDiT because
    dynamically padded text is laid out as ``[real text, pad text, image tokens]``;
    a simple sequence length would incorrectly mask image tokens after the text padding.
    """
    valid_tokens = valid_tokens.to(device=valid_tokens.device, dtype=torch.bool)
    B, length = valid_tokens.shape
    padded = ((length + block_size - 1) // block_size) * block_size
    if padded != length:
        valid_tokens = torch.nn.functional.pad(valid_tokens, (0, padded - length), value=False)

    def mask_mod(b, h, q_idx, kv_idx):
        return valid_tokens[b, q_idx] & valid_tokens[b, kv_idx]

    return create_block_mask(
        mask_mod, B=B, H=num_heads, Q_LEN=padded, KV_LEN=padded,
        device=valid_tokens.device, BLOCK_SIZE=block_size, _compile=_compile,
    )


def dense_document_mask(seq_lens, device):
    """SDPA fallback: (B,1,L,L) additive mask, 0 allowed / -inf disallowed. For tests/correctness."""
    max_len = max(seq_lens)
    B = len(seq_lens)
    m = torch.full((B, 1, max_len, max_len), float("-inf"), device=device)
    for i, n in enumerate(seq_lens):
        m[i, 0, :n, :n] = 0.0
    return m


def flex_attn(q, k, v, block_mask):
    """flex_attention must run COMPILED (eager gives wrong results per torch warning)."""
    if torch.compiler.is_compiling():
        return flex_attention(q, k, v, block_mask=block_mask)
    if _HAS_COMPILED:
        return _compiled_flex_attention(q, k, v, block_mask=block_mask)
    if not hasattr(flex_attn, "_fn"):
        flex_attn._fn = torch.compile(flex_attention)
    return flex_attn._fn(q, k, v, block_mask=block_mask)
