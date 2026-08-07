"""MMDiT = diffusers Flux2Transformer2DModel wrapper.

Dual-stream = num_layers, single-stream = num_single_layers (ablation: num_layers=0 = pure
single-stream). it2i editing via the native ref stream: ref latents are prepended to
hidden_states and counted with num_ref_tokens (ref tokens held at a clean fixed timestep).
t2i = num_ref_tokens=0.

Velocity target = noise - clean (flow.py convention, t=0 clean -> t=1 noise).
"""
from __future__ import annotations

import os
import torch
from torch import nn
import torch.nn.functional as F

from diffusers import Flux2Transformer2DModel

from ..config import ComponentConfig, ModelConfig
from .flow import interpolate, logit_normal_timesteps, velocity_target, euler_schedule, euler_step
from .vae import patchify_latents, unpatchify_latents


class _PerTokenAdaLayerNormContinuous(nn.Module):
    """AdaLayerNormContinuous compatible with scalar or per-token conditions.

    Keeping this as one module is important for FSDP2: its pre-forward hook
    converts both inputs and parameters consistently before this method runs.
    The child names match diffusers, so checkpoint keys remain unchanged.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.silu = base.silu
        self.linear = base.linear
        self.norm = base.norm

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if conditioning_embedding.ndim == 2:
            scale, shift = torch.chunk(emb, 2, dim=1)
            return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        if conditioning_embedding.ndim == 3:
            scale, shift = torch.chunk(emb, 2, dim=-1)
            return self.norm(x) * (1 + scale) + shift
        raise ValueError(
            "conditioning_embedding must be (B,D) or (B,N,D), got "
            f"{tuple(conditioning_embedding.shape)}"
        )


def _positions(batch: int, h: int, w: int, device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    return torch.stack((torch.zeros_like(x), y, x, torch.zeros_like(x)), -1).reshape(1, -1, 4).expand(batch, -1, -1)


def _text_positions(batch: int, length: int, device) -> torch.Tensor:
    pos = torch.zeros(batch, length, 4, device=device, dtype=torch.long)
    pos[..., 3] = torch.arange(length, device=device)
    return pos


class MMDiT(nn.Module):
    """Wrapper around Flux2Transformer2DModel."""

    def __init__(self, transformer: Flux2Transformer2DModel, in_channels: int, latent_channels: int, flex_attention: bool = False):
        super().__init__()
        transformer.norm_out = _PerTokenAdaLayerNormContinuous(transformer.norm_out)
        self.transformer = transformer
        self.in_channels = in_channels            # 128 = latent_ch * 4 (packed)
        self.latent_channels = latent_channels   # 32
        self.inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        self._flex = flex_attention
        self.compute_dtype: torch.dtype | None = None
        if not transformer.transformer_blocks:
            # Diffusers still constructs double-stream modulation weights for a
            # pure-single model, but no forward path can reach them. Keeping
            # them trainable creates empty/zero Adam state only after resume and
            # breaks optimizer-state equivalence without changing the model.
            for module in (
                transformer.double_stream_modulation_img,
                transformer.double_stream_modulation_txt,
            ):
                module.requires_grad_(False)
        if flex_attention:
            from .flex_mmdit import set_flex_attn
            set_flex_attn(self, enable=True)

    @classmethod
    def build(cls, component: ComponentConfig, cfg: ModelConfig, text_encoder=None):
        if component.backend != "random":
            raise ValueError("MMDiT build() expects backend='random' (from-scratch)")
        head_dim = cfg.hidden_size // cfg.num_heads
        if cfg.hidden_size % cfg.num_heads or head_dim % 8:
            raise ValueError("hidden_size/num_heads must give head_dim divisible by 8 for RoPE")
        double_layers = cfg.num_double_layers if cfg.num_double_layers is not None else cfg.num_layers
        single_layers = cfg.num_single_layers if cfg.num_single_layers is not None else cfg.num_layers
        text_dim = text_encoder.hidden_size if text_encoder is not None else cfg.hidden_size
        transformer = Flux2Transformer2DModel(
            in_channels=cfg.latent_channels * 4,
            out_channels=cfg.latent_channels * 4,
            num_layers=double_layers,
            num_single_layers=single_layers,
            attention_head_dim=head_dim,
            num_attention_heads=cfg.num_heads,
            joint_attention_dim=text_dim,
            axes_dims_rope=(head_dim // 4,) * 4,
            rope_theta=cfg.rope_theta,
            guidance_embeds=cfg.guidance_embeds,
        )
        return cls(transformer, in_channels=cfg.latent_channels * 4, latent_channels=cfg.latent_channels,
                   flex_attention=getattr(cfg, "flex_attention", False))

    @property
    def double_blocks(self):
        return self.transformer.transformer_blocks

    @property
    def single_blocks(self):
        return self.transformer.single_transformer_blocks

    @property
    def dtype(self):
        return self.compute_dtype or next(self.transformer.parameters()).dtype

    def forward(self, noisy, text, timesteps, **kwargs):
        """Compiled training entry point; delegates to velocity prediction."""
        return self.predict_velocity(noisy, text, timesteps, **kwargs)

    def forward_per_token(
        self,
        hidden_states: torch.Tensor,      # (B, N_ref+N_img, in_channels)
        encoder_hidden_states: torch.Tensor,  # (B, N_txt, joint_dim)
        sample_t: torch.Tensor,           # (B,) scalar t for text-stream modulation
        token_timesteps: torch.Tensor,    # (B, N_ref+N_img) per-token t for img modulation
        img_ids: torch.Tensor,            # (N_img_pos, 4) or (B, N, 4)
        txt_ids: torch.Tensor,            # (N_txt_pos, 4) or (B, N, 4)
        num_ref_tokens: int = 0,
        text_attn_mask: torch.Tensor | None = None,  # (B, 1, 1, L_kv) additive or None
    ) -> torch.Tensor:
        """Orchestrate Flux2 submodules with per-token timestep modulation (UniWorld self-flow).

        Instead of calling transformer.forward(), we directly invoke submodules so that img-stream
        modulation can be per-token (token_timesteps) while txt-stream stays per-sample (sample_t).
        This is possible because Flux2Modulation is SiLU+Linear on the last dim — feed (B,N,D) → (B,N,3D).
        """
        t = self.transformer
        B = hidden_states.shape[0]
        mdtype = self.dtype
        n_txt = encoder_hidden_states.shape[1]

        # build joint_attention_kwargs with mask if provided
        jakw = {"attention_mask": text_attn_mask} if text_attn_mask is not None else {}

        # --- 1. Input projections ---
        hidden_states = t.x_embedder(hidden_states)                    # (B, N, inner_dim)
        encoder_hidden_states = t.context_embedder(encoder_hidden_states.to(mdtype))

        # --- 2. RoPE ---
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = t.pos_embed(img_ids)
        text_rotary_emb = t.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # --- 3. Modulation ---
        # txt: per-sample (scalar t)
        temb_scalar = t.time_guidance_embed(sample_t.to(mdtype) * 1000, None)   # (B, D)
        # img: per-token (token_timesteps)
        tt_flat = (token_timesteps.to(mdtype).reshape(-1) * 1000)              # (B*N,)
        temb_tt = t.time_guidance_embed(tt_flat, None)                         # (B*N, D)
        temb_tt = temb_tt.reshape(B, -1, temb_scalar.shape[-1])               # (B, N, D)

        # A pure-single model has no consumer for the double-stream modulation
        # tensors. Besides wasting work, calling these frozen FSDP units with
        # no-reshard leaves their exposed parameter view in BF16 because no
        # backward hook runs to restore the FP32 shard. Skip the unreachable
        # modules entirely; mixed double+single models keep the original path.
        double_mod_txt = double_mod_img = None
        if t.transformer_blocks:
            double_mod_txt = t.double_stream_modulation_txt(temb_scalar)        # (B, 6D)
            double_mod_img = t.double_stream_modulation_img(temb_tt)            # (B, N, 6D)

        # single: txt scalar + img per-token, concatenated
        single_mod_scalar = t.single_stream_modulation(temb_scalar)            # (B, 3D)
        single_mod_tokens = t.single_stream_modulation(temb_tt)                # (B, N, 3D)
        single_mod = torch.cat([
            single_mod_scalar.unsqueeze(1).expand(B, n_txt, -1),
            single_mod_tokens,
        ], dim=1)                                                              # (B, N_txt+N, 3D)

        # --- 4. Double-stream blocks ---
        for block in t.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_mod_img,
                temb_mod_txt=double_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=jakw,
            )

        # --- 5. Concatenate for single stream [txt, img] ---
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # --- 6. Single-stream blocks (encoder_hidden_states=None: already concatenated) ---
        for block in t.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,   # already concatenated above
                temb_mod=single_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=jakw,
            )

        # --- 7. Output: drop txt/ref, keep target image tokens ---
        out = hidden_states[:, n_txt + num_ref_tokens:]                      # (B, N_img, D)
        # Self-Flow retains heterogeneous token timesteps through the output
        # head too, matching UniWorld's per-token FinalLayer.
        out_temb = temb_tt[:, num_ref_tokens:]
        out = t.norm_out(out, out_temb)
        out = t.proj_out(out)                                                 # (B, N_img, in_channels)
        return out

    def predict_velocity(
        self,
        noisy: torch.Tensor,           # (B, C, H, W) normalized latents
        text: torch.Tensor,            # (B, Ltxt, joint_dim) raw text feats
        timesteps: torch.Tensor,        # (B,) scalar t
        num_ref_tokens: int = 0,
        ref: torch.Tensor | None = None,  # (B, C, Hr, Wr) normalized ref latents (it2i)
        token_timesteps: torch.Tensor | None = None,  # (B, N_ref+N_img) per-token t (self-flow)
        text_attn_mask: torch.Tensor | None = None,  # (B, 1, 1, L_kv) additive mask for pad tokens
    ) -> torch.Tensor:
        b, c, h, w = noisy.shape
        mdtype = self.dtype
        packed = patchify_latents(noisy.to(mdtype))
        ph, pw = packed.shape[-2], packed.shape[-1]
        img_tokens = packed.flatten(2).transpose(1, 2)

        if ref is not None and num_ref_tokens > 0:
            ref_packed = patchify_latents(ref.to(mdtype))
            ref_tokens = ref_packed.flatten(2).transpose(1, 2)
            hidden = torch.cat([ref_tokens, img_tokens], dim=1)
            ref_ph, ref_pw = ref_packed.shape[-2], ref_packed.shape[-1]
            img_ids = torch.cat(
                [_positions(b, ref_ph, ref_pw, noisy.device), _positions(b, ph, pw, noisy.device)],
                dim=1,
            )
        else:
            hidden = img_tokens
            num_ref_tokens = 0
            img_ids = _positions(b, ph, pw, noisy.device)

        n_txt = text.shape[1]

        # --- per-token path (self-flow): orchestrate submodules directly ---
        if token_timesteps is not None:
            # if ref tokens exist, prepend t=0 (clean) for ref segment
            if num_ref_tokens > 0:
                ref_t = torch.zeros(b, num_ref_tokens, device=noisy.device, dtype=mdtype)
                token_timesteps = torch.cat([ref_t, token_timesteps], dim=1)
            out = self.forward_per_token(
                hidden_states=hidden,
                encoder_hidden_states=text,
                sample_t=timesteps,
                token_timesteps=token_timesteps,
                img_ids=img_ids,
                txt_ids=_text_positions(b, n_txt, noisy.device),
                num_ref_tokens=num_ref_tokens,
                text_attn_mask=text_attn_mask,
            )
            pred = out.transpose(1, 2).reshape_as(packed)
            return unpatchify_latents(pred)

        # --- scalar path (original, zero regression) ---
        ref_fixed_t = torch.zeros(b, device=noisy.device, dtype=mdtype) if num_ref_tokens > 0 else None
        fwd_kwargs = dict(
            hidden_states=hidden,
            encoder_hidden_states=text.to(mdtype),
            timestep=timesteps.to(mdtype),
            img_ids=img_ids,
            txt_ids=_text_positions(b, n_txt, noisy.device),
            guidance=None,
        )
        if text_attn_mask is not None:
            fwd_kwargs["joint_attention_kwargs"] = {"attention_mask": text_attn_mask}
        # only pass the ref-stream args when actually doing it2i (t2i: omit, matching UniWorld)
        if num_ref_tokens > 0 and ref is not None:
            fwd_kwargs["num_ref_tokens"] = num_ref_tokens
            fwd_kwargs["ref_fixed_timestep"] = ref_fixed_t
        if os.environ.get("UV3_DEBUG"):
            print(f"[dbg] hidden={tuple(hidden.shape)} text={tuple(text.shape)} "
                  f"img_ids={tuple(_positions(b, ph, pw, noisy.device).shape)} "
                  f"txt_ids={tuple(_text_positions(b, n_txt, noisy.device).shape)} "
                  f"mdtype={mdtype}", flush=True)
        if self._flex and text_attn_mask is None:
            from .flex_mmdit import set_block_mask
            from ..train.flex_attn import build_document_block_mask
            # bidirectional document mask: each batch element attends its own [txt+img] (no cross-sample)
            # NOTE: batched + per-batch timestep -> no packing; mask is full-attend (flex runs, ~SDPA speed).
            # Real speedup needs per-token timestep + packing (future).
            H = getattr(self.transformer.config, "num_attention_heads", 1)
            # document length = txt + (ref+img for it2i, or img for t2i); hidden already includes ref
            seq_lens = [n_txt + hidden.shape[1]] * b
            bm = build_document_block_mask(seq_lens, H, noisy.device, block_size=128, _compile=False)
            set_block_mask(self, bm)
        out = self.transformer(**fwd_kwargs).sample
        # drop ref tokens from output, keep only img
        out = out[:, num_ref_tokens:] if num_ref_tokens > 0 else out
        pred = out.transpose(1, 2).reshape_as(packed)
        return unpatchify_latents(pred)

    def training_loss(
        self,
        clean: torch.Tensor,           # (B, C, H, W) normalized latents
        text: torch.Tensor,            # (B, Ltxt, inner_dim)
        timesteps: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
        num_ref_tokens: int = 0,
        text_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = clean.shape[0]
        if timesteps is None:
            timesteps = logit_normal_timesteps(b, clean.device)
        noise = torch.randn_like(clean)
        noisy = interpolate(clean, noise, timesteps)
        pred_v = self(
            noisy, text, timesteps,
            num_ref_tokens=num_ref_tokens, ref=ref, text_attn_mask=text_attn_mask,
        )
        return F.mse_loss(pred_v.float(), velocity_target(clean, noise).float())

    @torch.no_grad()
    def sample_latents(self, shape, text, steps=30, device=None, dtype=None,
                       text_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        dtype = dtype or self.dtype
        latents = torch.randn(shape, device=device, dtype=dtype)
        times = euler_schedule(steps, device, dtype)
        for current, following in zip(times[:-1], times[1:]):
            v = self.predict_velocity(latents, text, current.expand(shape[0]),
                                      text_attn_mask=text_attn_mask)
            latents = euler_step(latents, v, current, following)
        return latents
