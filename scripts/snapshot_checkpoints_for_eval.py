#!/usr/bin/env python3
"""Persist atomic training checkpoints for asynchronous per-step evaluation."""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from uv3.train.fsdp2 import resolve_checkpoint_path


def checkpoint_step(path: Path) -> int:
    payload = torch.load(
        resolve_checkpoint_path(path), map_location="cpu", mmap=True, weights_only=False,
    )
    return int(payload.get("step", -1))


def snapshot(source: Path, destination: Path, expected_step: int) -> None:
    source = resolve_checkpoint_path(source)
    if destination.exists():
        actual = checkpoint_step(destination)
        if actual != expected_step:
            raise RuntimeError(
                f"existing snapshot step mismatch: {destination}={actual}, expected={expected_step}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary snapshot already exists: {temporary}")
    source_size = source.stat().st_size
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source_size:
            raise RuntimeError(
                f"snapshot size mismatch: {temporary.stat().st_size} != {source_size}"
            )
        actual = checkpoint_step(temporary)
        if actual != expected_step:
            raise RuntimeError(f"snapshot step mismatch: {actual} != {expected_step}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-step", type=int, default=10_000)
    args = parser.parse_args()

    source = args.run_dir / "ckpt.pt"
    seen_signature: tuple[int, int] | None = None
    while True:
        try:
            stat = source.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != seen_signature:
                step = checkpoint_step(source)
                if step <= 0:
                    raise RuntimeError(f"invalid checkpoint step={step}: {source}")
                destination = args.snapshot_dir / f"ckpt-step{step:08d}.pt"
                snapshot(source, destination, step)
                seen_signature = signature
                print(f"[snapshot] step={step} path={destination}", flush=True)
                if step >= args.max_step:
                    print(f"[snapshot] reached max_step={args.max_step}; done", flush=True)
                    return
        except FileNotFoundError:
            pass
        except Exception as error:
            print(f"[snapshot] retry after {type(error).__name__}: {error}", flush=True)
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
