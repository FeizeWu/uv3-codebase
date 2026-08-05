"""Self-flow matching (BFL Self-Flow), ported to continuous rectified-flow.

UniWorld third_party/DiT/train.py self-flow is built on DISCRETE DDPM timesteps
(q_sample/num_timesteps/timestep_to_clean_fraction). Here we use CONTINUOUS t in [0,1]
(t=0 clean, t=1 noise, flow.py convention), so the latent construction is re-derived:
- teacher timestep t_teacher = t_student * ratio  (closer to clean, smaller t)
- student latents use per-token mixing between student(t) and paired(t*?) tokens
- EMA teacher forward at t_teacher (no grad) -> capture block-N features
- student forward -> capture block-M features (M<N) -> project -> cosine align to teacher

Reusable from UniWorld as-is: self_flow_feature_loss (cosine), build_self_flow_projector
(Linear->SiLU->Linear). Teacher feature capture: forward hook on Flux2 block-N (diffusers
Flux2Transformer2DModel has no take_self_flow_outputs()).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_self_flow_projector(hidden_size: int, projector_dim: int | None = None) -> nn.Module:
    """BFL Self-Flow SimpleHead: Linear -> SiLU -> Linear."""
    if projector_dim is None:
        projector_dim = hidden_size * 2
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, hidden_size),
    )


def self_flow_feature_loss(teacher_tokens: torch.Tensor, student_tokens: torch.Tensor) -> torch.Tensor:
    """Cosine distance; teacher detached (no grad to teacher)."""
    teacher = F.normalize(teacher_tokens.float().detach(), dim=-1)
    student = F.normalize(student_tokens.float(), dim=-1)
    return 1.0 - (teacher * student).sum(dim=-1).mean()


class FeatureCapture:
    """Forward-hook capture of a block's output hidden states (for self-flow teacher/student).

    For Flux2 DOUBLE blocks the forward returns (txt_stream, img_stream) -> stream=1 takes img.
    For Flux2 SINGLE blocks the forward returns a single tensor = cat([txt, img]); img is the
    tail (after n_txt tokens). Use stream='img_double' / 'img_single_tail' / 'all'.
    """

    def __init__(self, stream: str = "all", n_txt: int = 0):
        self.features = None
        self._handles = []
        self.stream = stream
        self.n_txt = n_txt

    def attach(self, module: nn.Module):
        def hook(_m, _i, out):
            if self.stream == "img_double":
                # double block returns (txt, img) -> take img stream
                self.features = out[1] if isinstance(out, tuple) else out
            elif self.stream == "img_single_tail":
                # single block returns cat([txt, img]); img is tail
                t = out[0] if isinstance(out, tuple) else out
                self.features = t[:, self.n_txt:] if self.n_txt > 0 else t
            else:
                if hasattr(out, "hidden_states"):
                    self.features = out.hidden_states
                elif isinstance(out, tuple):
                    self.features = out[0]
                else:
                    self.features = out
        self._handles.append(module.register_forward_hook(hook, always_call=True))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def attach_self_flow_feature_captures(
    student: nn.Module,
    teacher: nn.Module,
    student_depth: int = -1,
    teacher_depth: int = -1,
) -> tuple[FeatureCapture, FeatureCapture, bool]:
    """Attach Self-Flow hooks for dual/single-stream and pure-single MMDiTs.

    The default student target is the first available block, while the teacher
    target is the last single-stream block. A pure-single student capture still
    contains the leading text tokens, so the returned boolean tells the trainer
    to remove that prefix before applying the projector/loss.
    """
    student_double = list(student.double_blocks)
    student_single = list(student.single_blocks)
    teacher_single = list(teacher.single_blocks)
    if not teacher_single:
        raise ValueError("Self-Flow requires at least one teacher single-stream block")

    if student_double:
        index = 0 if student_depth < 0 else student_depth
        if not -len(student_double) <= index < len(student_double):
            raise ValueError(
                f"student_depth={student_depth} is out of range for "
                f"{len(student_double)} double-stream blocks"
            )
        student_capture = FeatureCapture(stream="img_double")
        student_capture.attach(student_double[index])
        student_has_text_prefix = False
    else:
        if not student_single:
            raise ValueError("Self-Flow requires at least one student transformer block")
        index = 0 if student_depth < 0 else student_depth
        if not -len(student_single) <= index < len(student_single):
            raise ValueError(
                f"student_depth={student_depth} is out of range for "
                f"{len(student_single)} single-stream blocks"
            )
        student_capture = FeatureCapture(stream="all")
        student_capture.attach(student_single[index])
        student_has_text_prefix = True

    teacher_index = -1 if teacher_depth < 0 else teacher_depth
    if not -len(teacher_single) <= teacher_index < len(teacher_single):
        raise ValueError(
            f"teacher_depth={teacher_depth} is out of range for "
            f"{len(teacher_single)} single-stream blocks"
        )
    teacher_capture = FeatureCapture(stream="all")
    teacher_capture.attach(teacher_single[teacher_index])
    return student_capture, teacher_capture, student_has_text_prefix


def build_self_flow_latents_continuous(
    clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    mask_ratio: float = 0.5, ratio: float = 0.5, patch_size: int = 2,
    timestep_mode: str = "ratio",
):
    """Continuous-t self-flow with per-token timestep mixing (UniWorld train.py:297-438 ported).

    Returns (student_latents, teacher_latents, teacher_t, student_token_timesteps).
    - student_latents: per-token mix of interpolate(clean,noise,t) [mask] and interpolate(clean,noise,paired_t) [~mask].
    - student_token_timesteps: (B, N_tok) per-token t = where(mask, t, paired_t) — for forward_per_token.
    - teacher_latents: interpolate(clean, noise, teacher_t) (cleaner, for EMA teacher fwd).
    - teacher_t: (B,) scalar teacher timestep.

    timestep_mode: "ratio" (paired_t = t*ratio), "random_cleaner" (paired_t = t*U(0,1)),
                   "min" (paired_t ~ U(0,1), teacher_t = min(t, paired_t)).
    """
    b, c, h, w = clean.shape
    from .flow import interpolate

    # --- timestep pairing (UniWorld train.py:297-312, continuous t) ---
    if timestep_mode == "ratio":
        paired_t = (t * ratio).clamp(0.0, 1.0)
        teacher_t = paired_t
    elif timestep_mode == "random_cleaner":
        paired_t = (t * torch.rand_like(t)).clamp(0.0, 1.0)
        teacher_t = paired_t
    elif timestep_mode == "min":
        paired_t = torch.rand_like(t)
        teacher_t = torch.minimum(t, paired_t)
    else:
        raise ValueError(f"unknown timestep_mode: {timestep_mode}")

    student_full = interpolate(clean, noise, t)            # at student t
    paired = interpolate(clean, noise, paired_t)           # at paired (cleaner)
    teacher = interpolate(clean, noise, teacher_t)         # at teacher (cleanest)

    # --- per-token mask in patchified space (UniWorld train.py:337-347) ---
    ph, pw = h // patch_size, w // patch_size
    n_tok = ph * pw
    token_mask = (torch.rand(b, n_tok, 1, device=clean.device) < mask_ratio)  # True -> student-t

    # per-token timesteps: where(mask, t, paired_t) — (B, N_tok)
    student_token_timesteps = torch.where(
        token_mask.squeeze(-1),
        t.unsqueeze(1).expand(b, n_tok),
        paired_t.unsqueeze(1).expand(b, n_tok),
    )

    from .vae import patchify_latents, unpatchify_latents
    if mask_ratio <= 0:
        # no mixing: student gets full t (not paired), token_timesteps all = t
        return student_full, teacher, teacher_t, t.unsqueeze(1).expand(b, n_tok)
    s_tok = patchify_latents(student_full)
    p_tok = patchify_latents(paired)
    mask = token_mask.permute(0, 2, 1).reshape(b, 1, ph, pw)
    mixed = torch.where(mask, s_tok, p_tok)
    student_latents = unpatchify_latents(mixed)
    return student_latents, teacher, teacher_t, student_token_timesteps
