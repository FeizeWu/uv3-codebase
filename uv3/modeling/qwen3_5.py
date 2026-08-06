"""Qwen3.5 dense models as frozen text-feature encoders for the MMDiT.

Loads Qwen3_5ForConditionalGeneration, keeps only the language_model (text backbone,
forwards (input_ids, attention_mask) -> last_hidden_state, projected to
the MMDiT inner_dim by a trainable bridge. The vision/understanding head is NOT used
this phase (efficiency-verification period; no understanding loss).
"""
from __future__ import annotations

import torch
from torch import nn


class Qwen3_5TextEncoder(nn.Module):
    def __init__(self, language_model, tokenizer, hidden_size: int, max_length: int):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, path: str, max_length: int = 1024, dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        full = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        # Qwen3_5ForConditionalGeneration: text backbone lives at model.language_model
        lm = getattr(getattr(full, "model", full), "language_model", None)
        if lm is None:
            lm = getattr(full, "language_model", None)
        if lm is None:
            raise RuntimeError("could not locate language_model on Qwen3.5 model")
        # free vision tower + lm_head memory (not needed for frozen text encoding)
        try:
            del full
        except Exception:
            pass
        torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return cls(lm, tok, lm.config.hidden_size, max_length)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) ids+mask -> (B, L, hidden_size) last_hidden_state."""
        out = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        return out.last_hidden_state

    def tokenize(self, texts, device, max_length: int | None = None):
        if isinstance(texts, str):
            texts = [texts]
        max_length = self.max_length if max_length is None else int(max_length)
        enc = self.tokenizer(
            texts,
            # Static compile shape: every batch is exactly its configured bucket
            # length. The attention mask keeps padded tokens semantically inert.
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)


class Qwen3_5EmbeddingBridge(nn.Module):
    """Trainable projection from Qwen3.5 hidden size to MMDiT inner_dim."""

    def __init__(self, encoder: Qwen3_5TextEncoder, output_dim: int):
        super().__init__()
        self.encoder = encoder            # frozen (no grad)
        self.projection = nn.Linear(encoder.hidden_size, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.encode_text(input_ids, attention_mask)
        return self.projection(feats)

    @property
    def hidden_size(self) -> int:
        return self.output_dim
