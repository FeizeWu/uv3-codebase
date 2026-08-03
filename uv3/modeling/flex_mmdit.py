"""FlexAttention processors for the Flux2 MMDiT (double + single blocks).

Mirrors diffusers Flux2AttnProcessor / Flux2ParallelSelfAttnProcessor but replaces the
SDPA dispatch with flex_attention + a BIDIRECTIONAL document BlockMask (set on .block_mask
before forward). The MMDiT is bidirectional (no causal); the mask only separates packed samples
+ padding. At 256px batched the speedup is marginal (short seqs); the real win is packing +
high-res. Wiring this makes the MMDiT flex-capable (flex actually runs, not just a utility).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
    _get_qkv_projections,
)
from diffusers.models.embeddings import apply_rotary_emb

from ..train.flex_attn import flex_attn


def _flex_padded(q, k, v, block_mask):
    """flex_attention requires q_len == BlockMask Q_LEN (multiple of BLOCK_SIZE=128).
    Pad q/k/v to the BlockMask length, run flex, unpad the output."""
    target = block_mask.shape[-1]
    orig = q.shape[2]
    pad = target - orig
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    out = flex_attn(q, k, v, block_mask)
    return out[:, :, :orig] if pad > 0 else out


class FlexFlux2AttnProcessor(Flux2AttnProcessor):
    """Double-stream block attention via flex_attention + bidirectional BlockMask."""

    def __init__(self):
        super().__init__()
        self.block_mask = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            eq = eq.unflatten(-1, (attn.heads, -1))
            ek = ek.unflatten(-1, (attn.heads, -1))
            ev = ev.unflatten(-1, (attn.heads, -1))
            eq = attn.norm_added_q(eq)
            ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], dim=1)
            key = torch.cat([ek, key], dim=1)
            value = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        block_mask = attention_mask if isinstance(attention_mask, BlockMask) else self.block_mask
        if block_mask is not None:
            hs = _flex_padded(q, k, v, block_mask)
        else:
            hs = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        hs = hs.transpose(1, 2).flatten(2, 3).to(query.dtype)
        ehs = None
        if encoder_hidden_states is not None:
            ehs, hs = hs.split_with_sizes(
                [encoder_hidden_states.shape[1], hs.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            ehs = attn.to_add_out(ehs)
        hs = attn.to_out[0](hs)
        hs = attn.to_out[1](hs)
        return (hs, ehs) if ehs is not None else hs


class FlexFlux2SingleAttnProcessor(Flux2ParallelSelfAttnProcessor):
    """Single-stream (parallel self-attn) via flex + bidirectional BlockMask."""

    def __init__(self):
        super().__init__()
        self.block_mask = None

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        block_mask = attention_mask if isinstance(attention_mask, BlockMask) else self.block_mask
        if block_mask is not None:
            hs = _flex_padded(q, k, v, block_mask)
        else:
            hs = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        hs = hs.transpose(1, 2).flatten(2, 3).to(query.dtype)
        mlp_hidden = attn.mlp_act_fn(mlp_hidden)
        hs = torch.cat([hs, mlp_hidden], dim=-1)
        return attn.to_out(hs)


def set_flex_attn(mmdit, enable: bool, block_mask=None):
    """Swap the Flux2 attn processors for flex variants (block_mask set per-forward)."""
    t = getattr(mmdit, "transformer", mmdit)
    procs = []
    if enable:
        dbl = getattr(t, "transformer_blocks", [])
        sgl = getattr(t, "single_transformer_blocks", [])
        for blk in dbl:
            p = FlexFlux2AttnProcessor()
            p.block_mask = block_mask
            blk.attn.set_processor(p)
        for blk in sgl:
            p = FlexFlux2SingleAttnProcessor()
            p.block_mask = block_mask
            blk.attn.set_processor(p)
    else:
        for blk in getattr(t, "transformer_blocks", []):
            blk.attn.set_processor(Flux2AttnProcessor())
        for blk in getattr(t, "single_transformer_blocks", []):
            blk.attn.set_processor(Flux2ParallelSelfAttnProcessor())


def set_block_mask(mmdit, block_mask):
    """Update the block_mask on all flex processors (call per-forward before transformer fwd)."""
    t = getattr(mmdit, "transformer", mmdit)
    for blk in getattr(t, "transformer_blocks", []):
        if hasattr(blk.attn.processor, "block_mask"):
            blk.attn.processor.block_mask = block_mask
    for blk in getattr(t, "single_transformer_blocks", []):
        if hasattr(blk.attn.processor, "block_mask"):
            blk.attn.processor.block_mask = block_mask
