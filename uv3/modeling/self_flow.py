"""Self-flow matching (BFL Self-Flow), using continuous rectified-flow timesteps.

Both timesteps are independently sampled from the training distribution. Here t is in
[0,1] with t=0 clean and t=1 noise, matching flow.py:
- teacher timestep t_teacher = min(t, s)
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
        self.global_depth: int | None = None

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
    student_depth: int | None = None,
    teacher_depth: int | None = None,
    student_depth_ratio: float = 0.3,
    teacher_depth_ratio: float = 0.7,
) -> tuple[FeatureCapture, FeatureCapture, bool, bool]:
    """Attach Self-Flow hooks for dual/single-stream and pure-single MMDiTs.

    Depths are global across double blocks followed by single blocks. Explicit
    indices override the paper's scale-independent defaults (0.3D student,
    0.7D teacher). The returned booleans tell the trainer whether each captured
    sequence contains the leading text-token prefix.
    """
    student_double = list(student.double_blocks)
    student_single = list(student.single_blocks)
    teacher_double = list(teacher.double_blocks)
    teacher_single = list(teacher.single_blocks)
    total_depth = len(student_double) + len(student_single)
    if total_depth < 2 or total_depth != len(teacher_double) + len(teacher_single):
        raise ValueError("Self-Flow requires matching student/teacher depths of at least two")

    def resolve_depth(explicit: int | None, ratio: float, label: str) -> int:
        if explicit is not None:
            index = explicit if explicit >= 0 else total_depth + explicit
        else:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{label}_depth_ratio must be in [0,1], got {ratio}")
            index = min(total_depth - 1, int(ratio * total_depth))
        if not 0 <= index < total_depth:
            raise ValueError(f"{label}_depth={explicit} resolves to {index}, depth={total_depth}")
        return index

    student_index = resolve_depth(student_depth, student_depth_ratio, "student")
    teacher_index = resolve_depth(teacher_depth, teacher_depth_ratio, "teacher")
    if student_index >= teacher_index:
        raise ValueError(
            f"Self-Flow requires student depth < teacher depth, got "
            f"{student_index} >= {teacher_index}"
        )

    def attach_at_depth(double_blocks, single_blocks, index):
        if index < len(double_blocks):
            capture = FeatureCapture(stream="img_double")
            capture.attach(double_blocks[index])
            has_text_prefix = False
        else:
            capture = FeatureCapture(stream="all")
            capture.attach(single_blocks[index - len(double_blocks)])
            has_text_prefix = True
        capture.global_depth = index
        return capture, has_text_prefix

    student_capture, student_has_text_prefix = attach_at_depth(
        student_double, student_single, student_index,
    )
    teacher_capture, teacher_has_text_prefix = attach_at_depth(
        teacher_double, teacher_single, teacher_index,
    )
    return (
        student_capture,
        teacher_capture,
        student_has_text_prefix,
        teacher_has_text_prefix,
    )


def build_self_flow_latents_continuous(
    clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    mask_ratio: float = 0.25, ratio: float = 0.5, patch_size: int = 2,
    timestep_mode: str = "independent", paired_t: torch.Tensor | None = None,
):
    """Continuous-t self-flow with per-token timestep mixing (UniWorld train.py:297-438 ported).

    Returns (student_latents, teacher_latents, teacher_t, student_token_timesteps).
    - student_latents: per-token mix of interpolate(clean,noise,t) [mask] and interpolate(clean,noise,paired_t) [~mask].
    - student_token_timesteps: (B, N_tok) per-token t = where(mask, t, paired_t) — for forward_per_token.
    - teacher_latents: interpolate(clean, noise, teacher_t) (cleaner, for EMA teacher fwd).
    - teacher_t: (B,) scalar teacher timestep.

    timestep_mode: "independent" samples t and s independently from the same
                   caller-provided training distribution and uses min(t,s) for
                   the teacher. The other modes are retained as explicit ablations.
    """
    b, c, h, w = clean.shape
    from .flow import interpolate

    # --- timestep pairing (UniWorld train.py:297-312, continuous t) ---
    if timestep_mode == "independent":
        if paired_t is None:
            raise ValueError("independent Self-Flow requires a caller-sampled paired_t")
        if paired_t.shape != t.shape:
            raise ValueError(f"paired_t shape {paired_t.shape} must match t shape {t.shape}")
        paired_t = paired_t.to(device=t.device, dtype=t.dtype).clamp(0.0, 1.0)
        teacher_t = torch.minimum(t, paired_t)
    elif timestep_mode == "ratio":
        paired_t = (t * ratio).clamp(0.0, 1.0)
        teacher_t = paired_t
    elif timestep_mode == "random_cleaner":
        paired_t = (t * torch.rand_like(t)).clamp(0.0, 1.0)
        teacher_t = paired_t
    elif timestep_mode == "min":
        paired_t = torch.rand_like(t) if paired_t is None else paired_t
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
