"""6-stage timing (data / vae / qwen / forward / loss / backward / optimizer)."""
from __future__ import annotations

import time

import torch


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = {}
        self._marks = []

    def mark(self, name: str):
        self._marks.append((name, time.time()))
        if name in self.t:
            self.t[name] += self._marks[-1][1] - self._marks[-2][1]
        else:
            self.t[name] = self._marks[-1][1] - self._marks[-2][1]

    def start(self):
        self._marks = [("start", time.time())]

    def stop(self, name: str = "total"):
        self.mark(name)

    def summary(self) -> dict:
        return {k: v for k, v in self.t.items() if k != "start"}


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
